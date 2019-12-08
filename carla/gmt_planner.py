import pycuda
from pycuda.scan import ExclusiveScanKernel
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as cuda
import pycuda.cumath as cumath
from pycuda.compiler import SourceModule

import numpy as np
import math
from timeit import default_timer as timer
from kernel import *

wavefront = mod.get_function("wavefront")
neighborIndicator = mod.get_function("neighborIndicator")
exclusiveScan = ExclusiveScanKernel(np.int32, "a+b", 0)
compact = mod.get_function("compact")
dubinConnection = mod.get_function("dubinConnection")

class GMT(object):
    def __init__(self, init_parameters, debug=False):
        self._cpu_init(init_parameters, debug)
        self._gpu_init(debug)

        self.route = []
        self.start = 0

    def _cpu_init(self, init_parameters, debug):
        self.states = init_parameters['states']
        self.n = self.states.shape[0]
        self.waypoints = np.arange(self.n).astype(np.int32)

        self.neighbors = init_parameters['neighbors']
        self.num_neighbors = init_parameters['num_neighbors']

        self.cost = np.full(self.n, np.inf).astype(np.float32)
        self.Vunexplored = np.full(self.n, 1).astype(np.int32)
        self.Vopen = np.zeros_like(self.Vunexplored).astype(np.int32)

        self.threadsPerBlock = init_parameters['threadsPerBlock']
        self.nBlocksPerGrid = int(((self.n + self.threadsPerBlock - 1) / self.threadsPerBlock))
        
        if debug:
            print('neighbors: ', self.neighbors)
            print('number neighbors: ', self.num_neighbors)

    def _gpu_init(self, debug):
        self.dev_states = cuda.to_gpu(self.states)
        self.dev_waypoints = cuda.to_gpu(self.waypoints)

        self.dev_n = cuda.to_gpu(np.array([self.n]).astype(np.int32))

        self.dev_neighbors = cuda.to_gpu(self.neighbors)
        self.dev_num_neighbors = cuda.to_gpu(self.num_neighbors)
        self.neighbors_index = cuda.to_gpu(self.num_neighbors)
        exclusiveScan(self.neighbors_index)

        self.dev_Gindicator = cuda.GPUArray(self.Vopen.shape,self.Vopen.dtype)

    def step_init(self, iter_parameters, debug):
        self.cost[self.start] = np.inf
        self.Vunexplored[self.start] = 1
        self.Vopen[self.start] = 0

        if self.start != iter_parameters['start'] and len(self.route) > 2:
            del self.route[-1]

        self.obstacles = iter_parameters['obstacles']
        self.num_obs = iter_parameters['num_obs']
        self.parent = np.full(self.n, -1).astype(np.int32)

        self.start = iter_parameters['start']
        self.goal = iter_parameters['goal']
        self.radius = iter_parameters['radius']
        self.threshold = np.array([ iter_parameters['threshold'] ]).astype(np.float32)

        self.cost[self.start] = 0
        self.Vunexplored[self.start] = 0
        self.Vopen[self.start] = 1

        if debug:
            print('parents:', self.parent)
            print('cost: ', self.cost)
            print('Vunexplored: ', self.Vunexplored)
            print('Vopen: ', self.Vopen)

        self.dev_radius = cuda.to_gpu(np.array([self.radius]).astype(np.float32))
        self.dev_threshold = cuda.to_gpu(self.threshold)

        self.dev_obstacles = cuda.to_gpu(self.obstacles) 
        self.dev_num_obs = cuda.to_gpu(self.num_obs)

        self.dev_parent = cuda.to_gpu(self.parent)
        self.dev_cost = cuda.to_gpu(self.cost)

        self.dev_open = cuda.to_gpu(self.Vopen)
        self.dev_unexplored = cuda.to_gpu(self.Vunexplored)

    def get_path(self):
        p = self.goal
        while p != -1:
            self.route.append(p)
            p = self.parent[p]

        # del self.route[-1]

    def run_step(self, iter_parameters, iter_limit=1000, debug=False):
        start_mem = timer() ############################# timer
        self.step_init(iter_parameters,debug)
        end_mem = timer() ############################# timer

        # print("memory time: ", end-start)    

        goal_reached = False
        iteration = 0
        while True:
            start_iter = timer() ############################# timer
            iteration += 1

            ########## create Wave front ###############
            start_wave_f = timer() ############################# timer
            wavefront(self.dev_Gindicator, self.dev_open, self.dev_cost, self.dev_threshold, self.dev_n, block=(self.threadsPerBlock,1,1), grid=(self.nBlocksPerGrid,1))
            end_wave_f = timer() ############################# timer

            start_wave_c = timer() ############################# timer
            self.dev_threshold += 2*self.dev_radius
            goal_reached = self.dev_Gindicator[self.goal].get() == 1
            end_wave_c = timer() ############################# timer
            
            dev_Gscan = cuda.to_gpu(self.dev_Gindicator)
            exclusiveScan(dev_Gscan)
            dev_gSize = dev_Gscan[-1] + self.dev_Gindicator[-1]
            gSize = int(dev_gSize.get())

            if iteration >= iter_limit:
                print('### iteration limit ###', iteration)
                return self.route
            elif goal_reached:
                print('### goal reached ### ', iteration)
                self.parent = self.dev_parent.get()
                self.route =[]
                self.get_path()
                return self.route
            elif gSize == 0:
                print('### threshold skip ', iteration)
                continue

            dev_G = cuda.zeros(gSize, dtype=np.int32)
            compact(dev_G, dev_Gscan, self.dev_Gindicator, self.dev_waypoints, self.dev_n, block=(self.threadsPerBlock,1,1), grid=(self.nBlocksPerGrid,1))     
            

            ######### scan and compact open set to connect neighbors ###############
            start_open = timer() ############################# timer
            dev_yscan = cuda.to_gpu(self.dev_open)
            exclusiveScan(dev_yscan)
            dev_ySize = dev_yscan[-1] + self.dev_open[-1]
            ySize = int(dev_ySize.get())

            dev_y = cuda.zeros(ySize, dtype=np.int32)
            compact(dev_y, dev_yscan, self.dev_open, self.dev_waypoints, self.dev_n, block=(self.threadsPerBlock,1,1), grid=(self.nBlocksPerGrid,1))
            end_open = timer() ############################# timer

            ########## creating neighbors of wave front to connect open ###############
            start_neighbor = timer()  ############################# timer
            dev_xindicator = cuda.zeros_like(self.dev_open, dtype=np.int32)
            gBlocksPerGrid = int(((gSize + self.threadsPerBlock - 1) / self.threadsPerBlock))
            neighborIndicator(dev_xindicator, dev_G, self.dev_unexplored, self.dev_neighbors, self.dev_num_neighbors, self.neighbors_index, dev_gSize, block=(self.threadsPerBlock,1,1), grid=(gBlocksPerGrid,1))
            end_neighbor = timer() ############################# timer

            start_neighbor_c = timer()  ############################# timer
            dev_xscan = cuda.to_gpu(dev_xindicator)
            exclusiveScan(dev_xscan)
            dev_xSize = dev_xscan[-1] + dev_xindicator[-1]
            xSize = int(dev_xSize.get())

            if xSize == 0:
                print('### x skip')
                continue

            dev_x = cuda.zeros(xSize, dtype=np.int32)
            compact(dev_x, dev_xscan, dev_xindicator, self.dev_waypoints, self.dev_n, block=(self.threadsPerBlock,1,1), grid=(self.nBlocksPerGrid,1))
            end_neighbor_c = timer()  ############################# timer

            ######### connect neighbors ####################
            # # launch planning
            start_connect = timer() ############################# timer
            xBlocksPerGrid = int(((xSize + self.threadsPerBlock - 1) / self.threadsPerBlock))
            dubinConnection(self.dev_cost, self.dev_parent, dev_x, dev_y, self.dev_states, self.dev_open, self.dev_unexplored, dev_xSize, dev_ySize, self.dev_obstacles, self.dev_num_obs, self.dev_radius, block=(self.threadsPerBlock,1,1), grid=(xBlocksPerGrid,1))
            end_connect =timer() ############################# timer

            end_iter = timer() ############################# timer

            if debug:
                print('dev parents:', self.dev_parent)
                print('dev cost: ', self.dev_cost)
                print('dev unexplored: ', self.dev_unexplored)
                print('dev open: ', self.dev_open)
                print('dev threshold: ', self.dev_threshold, self.dev_radius)

                print('y size: ', ySize, 'y: ' , dev_y)
                print('G size: ', gSize, 'G: ', dev_G)

                print('x size: ', dev_xSize, 'x: ', dev_x)

            print('wave front timer: ', end_wave_f-start_wave_f)
            print('wave compact timer: ', end_wave_c-start_wave_c)
            print('open set timer: ', end_open-start_open)
            print('neighbor timer: ', end_neighbor-start_neighbor)
            print('neighbor compact timer: ', end_neighbor_c-start_neighbor_c)
            print('connection timer: ', end_connect-start_connect)
            iteration_time = end_iter-start_iter
            print(f'######### iteration: {iteration} iteration time: {iteration_time}')










class GMTmem(object):
    def __init__(self, init_parameters, debug=False):
        self.route = []
        self.start = 0

        self._cpu_init(init_parameters, debug)
        self._gpu_init(debug)


    def _cpu_init(self, init_parameters, debug):
        self.states = init_parameters['states']
        self.n = self.states.shape[0]
        self.waypoints = np.arange(self.n).astype(np.int32)

        self.neighbors = init_parameters['neighbors']
        self.num_neighbors = init_parameters['num_neighbors']

        self.cost = np.full(self.n, np.inf).astype(np.float32)
        self.Vunexplored = np.full(self.n, 1).astype(np.int32)
        self.Vopen = np.zeros_like(self.Vunexplored).astype(np.int32)

        self.threadsPerBlock = init_parameters['threadsPerBlock']
        self.nBlocksPerGrid = int(((self.n + self.threadsPerBlock - 1) / self.threadsPerBlock))
        
        if debug:
            print('neighbors: ', self.neighbors)
            print('number neighbors: ', self.num_neighbors)

    def _gpu_init(self, debug):
        self.dev_n = cuda.to_gpu(np.array([self.n]).astype(np.int32))

        self.neighbors_index = cuda.to_gpu(self.num_neighbors)
        exclusiveScan(self.neighbors_index)

        self.dev_num_neighbors = cuda.to_gpu(self.num_neighbors)

        self.dev_states = cuda.to_gpu(self.states)
        self.dev_waypoints = cuda.to_gpu(self.waypoints)

        self.dev_neighbors = cuda.to_gpu(self.neighbors)

        self.dev_Gindicator = cuda.GPUArray(self.Vopen.shape,self.Vopen.dtype)
        self.dev_xindicator = cuda.GPUArray(self.Vopen.shape,self.Vopen.dtype)

        self.dev_xindicator_zeros = cuda.GPUArray(self.Vopen.shape,self.Vopen.dtype)

        self.zero_val = np.zeros((), np.int32)

        self.dev_xindicator_zeros.fill(self.zero_val)


    def step_init(self, iter_parameters, debug):
        self.cost[self.start] = np.inf
        self.Vunexplored[self.start] = 1
        self.Vopen[self.start] = 0

        if self.start != iter_parameters['start'] and len(self.route) > 2:
            del self.route[-1]

        self.obstacles = iter_parameters['obstacles']
        self.num_obs = iter_parameters['num_obs']
        self.parent = np.full(self.n, -1).astype(np.int32)

        self.start = iter_parameters['start']
        self.goal = iter_parameters['goal']
        self.radius = iter_parameters['radius']
        self.threshold = np.array([ iter_parameters['threshold'] ]).astype(np.float32)

        self.cost[self.start] = 0
        self.Vunexplored[self.start] = 0
        self.Vopen[self.start] = 1

        if debug:
            print('parents:', self.parent)
            print('cost: ', self.cost)
            print('Vunexplored: ', self.Vunexplored)
            print('Vopen: ', self.Vopen)

        self.dev_open = cuda.to_gpu(self.Vopen)

        self.dev_threshold = cuda.to_gpu(self.threshold)

        self.dev_radius = cuda.to_gpu(np.array([self.radius]).astype(np.float32))
        self.dev_obstacles = cuda.to_gpu(self.obstacles) 
        self.dev_num_obs = cuda.to_gpu(self.num_obs)

        self.dev_parent = cuda.to_gpu(self.parent)
        self.dev_cost = cuda.to_gpu(self.cost)
        
        self.dev_unexplored = cuda.to_gpu(self.Vunexplored)


    def get_path(self):
        p = self.goal
        while p != -1:
            self.route.append(p)
            p = self.parent[p]

    def run_step(self, iter_parameters, iter_limit=1000, debug=False):
        self.step_init(iter_parameters,debug)

        goal_reached = False
        iteration = 0
        while True:
            iteration += 1

            start_iter = timer()
            ########## create Wave front ###############
            start_wave_f = timer() ############################# timer
            wavefront(self.dev_Gindicator, self.dev_open, self.dev_cost, self.dev_threshold, self.dev_n, block=(self.threadsPerBlock,1,1), grid=(self.nBlocksPerGrid,1))
            end_wave_f = timer() ############################# timer

            start_wave_c = timer() ############################# timer
            self.dev_threshold += 2* self.dev_radius
            goal_reached = self.dev_Gindicator[self.goal].get() == 1
            
            dev_Gscan = cuda.to_gpu(self.dev_Gindicator)
            exclusiveScan(dev_Gscan)
            

            dev_gSize = dev_Gscan[-1] + self.dev_Gindicator[-1]
            gSize = int(dev_gSize.get())

            if iteration >= iter_limit:
                print('### iteration limit ###', iteration)
                return self.route
            elif goal_reached:
                print('### goal reached ### ', iteration)
                self.parent = self.dev_parent.get()
                self.route = []
                self.get_path()
                return self.route
            elif gSize == 0:
                print('### threshold skip ', iteration)
                continue

            dev_G = cuda.GPUArray([gSize,],np.int32)
            #dev_G = cuda.zeros(gSize, dtype=np.int32)
            
            compact(dev_G, dev_Gscan, self.dev_Gindicator, self.dev_waypoints, self.dev_n, block=(self.threadsPerBlock,1,1), grid=(self.nBlocksPerGrid,1))    
            end_wave_c = timer() ############################# timer
            


            ######### scan and compact open set to connect neighbors ###############
            start_open = timer() ############################# timer
            dev_yscan = cuda.to_gpu(self.dev_open)
            exclusiveScan(dev_yscan)
            dev_ySize = dev_yscan[-1] + self.dev_open[-1]
            ySize = int(dev_ySize.get())

            #dev_y = cuda.zeros(ySize, dtype=np.int32)
            dev_y = cuda.GPUArray([ySize,],np.int32)
            compact(dev_y, dev_yscan, self.dev_open, self.dev_waypoints, self.dev_n, block=(self.threadsPerBlock,1,1), grid=(self.nBlocksPerGrid,1))
            end_open = timer() ############################# timer

            ########## creating neighbors of wave front to connect open ###############
            
            #dev_xindicator = cuda.zeros_like(self.dev_open, dtype= np.int32,stream= self.stream1)

            #self.dev_xindicator.fill(self.zero_val, stream = self.stream1)
            #print(self.dev_xindicator_zeros.nbytes)
            start_neighbor = timer()  ############################# timer
            drv.memcpy_dtod(self.dev_xindicator.gpudata,self.dev_xindicator_zeros.gpudata,self.dev_xindicator_zeros.nbytes)
            gBlocksPerGrid = int(((gSize + self.threadsPerBlock - 1) / self.threadsPerBlock))
            neighborIndicator(self.dev_xindicator, dev_G, self.dev_unexplored, self.dev_neighbors, self.dev_num_neighbors, self.neighbors_index, dev_gSize, block=(self.threadsPerBlock,1,1), grid=(gBlocksPerGrid,1))
            end_neighbor = timer() ############################# timer

            start_neighbor_c = timer()  ############################# timer
            dev_xscan = cuda.to_gpu(self.dev_xindicator)
            exclusiveScan(dev_xscan)


            #start_create_n= timer()
            #dev_xscan = cuda.to_gpu_async(self.dev_xindicator, stream=self.stream1)
            #start_create_n= timer()
            #dev_xSize = cuda.sum(self.dev_xindicator, stream = self.stream1)
            #exclusiveScan(dev_xscan, stream=self.stream1)
            #start_create_n= timer()
            dev_xSize = dev_xscan[-1] + self.dev_xindicator[-1]
            #end_create_n= timer()

            xSize = int(dev_xSize.get())
            
            if xSize == 0:
                print('### x skip')
                continue

            dev_x = cuda.GPUArray([xSize,],np.int32)
            #dev_x = cuda.zeros(xSize, dtype=np.int32)
            compact(dev_x, dev_xscan, self.dev_xindicator, self.dev_waypoints, self.dev_n, block=(self.threadsPerBlock,1,1), grid=(self.nBlocksPerGrid,1))
            end_neighbor_c = timer()  ############################# timer

            ######### connect neighbors ####################
            # # launch planning
            start_connect = timer() ############################# timer
            xBlocksPerGrid = int(((xSize + self.threadsPerBlock - 1) / self.threadsPerBlock))
            dubinConnection(self.dev_cost, self.dev_parent, dev_x, dev_y, self.dev_states, self.dev_open, self.dev_unexplored, dev_xSize, dev_ySize, self.dev_obstacles, self.dev_num_obs, self.dev_radius, block=(self.threadsPerBlock,1,1), grid=(xBlocksPerGrid,1))
            end_connect =timer() ############################# timer

            end_iter = timer()
            if debug:
                print('dev parents:', self.dev_parent)
                print('dev cost: ', self.dev_cost)
                print('dev unexplored: ', self.dev_unexplored)
                print('dev open: ', self.dev_open)
                print('dev threshold: ', self.dev_threshold)

                print('goal reached: ', goal_reached)
                print('y size: ', ySize, 'y: ' , dev_y)
                print('G size: ', gSize, 'G: ', dev_G)

                print('x size: ', dev_xSize, 'x: ', dev_x)
            print('wave front timer: ', end_wave_f-start_wave_f)
            print('wave compact timer: ', end_wave_c-start_wave_c)
            print('open set timer: ', end_open-start_open)
            print('neighbor timer: ', end_neighbor-start_neighbor)
            print('neighbor compact timer: ', end_neighbor_c-start_neighbor_c)
            print('connection timer: ', end_connect-start_connect)
            iteration_time = end_iter-start_iter
            print(f'######### iteration: {iteration} iteration time: {iteration_time}')















class GMTstream(object):
    def __init__(self, init_parameters, debug=False):
        self.route = []
        self.start = 0
        self.stream1 = drv.Stream()
        self.stream2 = drv.Stream()

        self._cpu_init(init_parameters, debug)
        self._gpu_init(debug)


    def _cpu_init(self, init_parameters, debug):
        self.states = init_parameters['states']
        self.n = self.states.shape[0]
        self.waypoints = np.arange(self.n).astype(np.int32)

        self.neighbors = init_parameters['neighbors']
        self.num_neighbors = init_parameters['num_neighbors']

        self.cost = np.full(self.n, np.inf).astype(np.float32)
        self.Vunexplored = np.full(self.n, 1).astype(np.int32)
        self.Vopen = np.zeros_like(self.Vunexplored).astype(np.int32)

        self.threadsPerBlock = init_parameters['threadsPerBlock']
        self.nBlocksPerGrid = int(((self.n + self.threadsPerBlock - 1) / self.threadsPerBlock))
        
        if debug:
            print('neighbors: ', self.neighbors)
            print('number neighbors: ', self.num_neighbors)

    def _gpu_init(self, debug):
        self.dev_n = cuda.to_gpu_async(np.array([self.n]).astype(np.int32), stream=self.stream2)

        self.neighbors_index = cuda.to_gpu_async(self.num_neighbors, stream=self.stream1)
        exclusiveScan(self.neighbors_index, stream=self.stream1)

        self.dev_num_neighbors = cuda.to_gpu_async(self.num_neighbors, stream=self.stream2)

        self.dev_states = cuda.to_gpu_async(self.states, stream=self.stream2)
        self.dev_waypoints = cuda.to_gpu_async(self.waypoints, stream=self.stream2)

        self.dev_neighbors = cuda.to_gpu_async(self.neighbors, stream=self.stream2)

        self.dev_Gindicator = cuda.GPUArray(self.Vopen.shape,self.Vopen.dtype)
        self.dev_xindicator = cuda.GPUArray(self.Vopen.shape,self.Vopen.dtype)

        self.dev_xindicator_zeros = cuda.GPUArray(self.Vopen.shape,self.Vopen.dtype)

        self.zero_val = np.zeros((), np.int32)

        self.dev_xindicator_zeros.fill(self.zero_val, stream = self.stream1)
        

        self.stream1.synchronize()
        self.stream2.synchronize()


    def step_init(self, iter_parameters, debug):
        self.cost[self.start] = np.inf
        self.Vunexplored[self.start] = 1
        self.Vopen[self.start] = 0

        if self.start != iter_parameters['start'] and len(self.route) > 2:
            del self.route[-1]

        self.obstacles = iter_parameters['obstacles']
        self.num_obs = iter_parameters['num_obs']
        self.parent = np.full(self.n, -1).astype(np.int32)

        self.start = iter_parameters['start']
        self.goal = iter_parameters['goal']
        self.radius = iter_parameters['radius']
        self.threshold = np.array([ iter_parameters['threshold'] ]).astype(np.float32)

        self.cost[self.start] = 0
        self.Vunexplored[self.start] = 0
        self.Vopen[self.start] = 1

        if debug:
            print('parents:', self.parent)
            print('cost: ', self.cost)
            print('Vunexplored: ', self.Vunexplored)
            print('Vopen: ', self.Vopen)

        self.dev_open = cuda.to_gpu_async(self.Vopen, stream=self.stream2)

        self.dev_threshold = cuda.to_gpu_async(self.threshold, stream=self.stream1)

        self.dev_radius = cuda.to_gpu_async(np.array([self.radius]).astype(np.float32), stream=self.stream2)
        self.dev_obstacles = cuda.to_gpu_async(self.obstacles, stream=self.stream2) 
        self.dev_num_obs = cuda.to_gpu_async(self.num_obs, stream=self.stream2)

        self.dev_parent = cuda.to_gpu_async(self.parent, stream=self.stream2)
        self.dev_cost = cuda.to_gpu_async(self.cost, stream=self.stream1)
        
        self.dev_unexplored = cuda.to_gpu_async(self.Vunexplored, stream=self.stream1)


    def get_path(self):
        p = self.goal
        while p != -1:
            self.route.append(p)
            p = self.parent[p]

    def run_step(self, iter_parameters, iter_limit=1000, debug=False):
        start = timer()
        self.step_init(iter_parameters,debug)
        end = timer()

        # print("memory time: ", end-start)  

        goal_reached = False
        iteration = 0
        while True:
            start_iter = timer()
            iteration += 1

            ########## create Wave front ###############
            wavefront(self.dev_Gindicator, self.dev_open, self.dev_cost, self.dev_threshold, self.dev_n, block=(self.threadsPerBlock,1,1), grid=(self.nBlocksPerGrid,1), stream=self.stream1)
            self.dev_threshold += 2*self.dev_radius
            goal_reached = self.dev_Gindicator[self.goal].get() == 1
            
            dev_Gscan = cuda.to_gpu_async(self.dev_Gindicator, stream=self.stream1)
            exclusiveScan(dev_Gscan, stream=self.stream1)
            dev_gSize = dev_Gscan[-1] + self.dev_Gindicator[-1]
            gSize = int(dev_gSize.get_async(stream=self.stream1))

            if iteration >= iter_limit:
                print('### iteration limit ###')
                return self.route
            elif goal_reached:
                print('### goal reached ### ', iteration)
                self.parent = self.dev_parent.get_async(stream=self.stream1)
                self.route = []
                self.get_path()
                return self.route
            elif gSize == 0:
                print('### threshold skip ', iteration)
                continue

            dev_G = cuda.zeros(gSize, dtype=np.int32)
            compact(dev_G, dev_Gscan, self.dev_Gindicator, self.dev_waypoints, self.dev_n, block=(self.threadsPerBlock,1,1), grid=(self.nBlocksPerGrid,1), stream=self.stream1)    

            ######### scan and compact open set to connect neighbors ###############
            dev_yscan = cuda.to_gpu_async(self.dev_open, stream=self.stream2)
            exclusiveScan(dev_yscan, stream=self.stream2)
            dev_ySize = dev_yscan[-1] + self.dev_open[-1]
            ySize = int(dev_ySize.get_async(stream=self.stream2))

            dev_y = cuda.zeros(ySize, dtype=np.int32)
            compact(dev_y, dev_yscan, self.dev_open, self.dev_waypoints, self.dev_n, block=(self.threadsPerBlock,1,1), grid=(self.nBlocksPerGrid,1), stream=self.stream2)

            ########## creating neighbors of wave front to connect open ###############
            dev_xindicator = cuda.zeros_like(self.dev_open, dtype=np.int32)
            gBlocksPerGrid = int(((gSize + self.threadsPerBlock - 1) / self.threadsPerBlock))
            neighborIndicator(dev_xindicator, dev_G, self.dev_unexplored, self.dev_neighbors, self.dev_num_neighbors, self.neighbors_index, dev_gSize, block=(self.threadsPerBlock,1,1), grid=(gBlocksPerGrid,1), stream=self.stream1)

            dev_xscan = cuda.to_gpu_async(dev_xindicator, stream=self.stream1)
            exclusiveScan(dev_xscan, stream=self.stream1)
            dev_xSize = dev_xscan[-1] + dev_xindicator[-1]
            xSize = int(dev_xSize.get_async(stream=self.stream1))

            if xSize == 0:
                print('### x skip')
                continue

            dev_x = cuda.zeros(xSize, dtype=np.int32)
            compact(dev_x, dev_xscan, dev_xindicator, self.dev_waypoints, self.dev_n, block=(self.threadsPerBlock,1,1), grid=(self.nBlocksPerGrid,1), stream=self.stream1)

            # self.stream1.synchronize()
            self.stream2.synchronize()

            ######### connect neighbors ####################
            # # launch planning
            xBlocksPerGrid = int(((xSize + self.threadsPerBlock - 1) / self.threadsPerBlock))
            dubinConnection(self.dev_cost, self.dev_parent, dev_x, dev_y, self.dev_states, self.dev_open, self.dev_unexplored, dev_xSize, dev_ySize, self.dev_obstacles, self.dev_num_obs, self.dev_radius, block=(self.threadsPerBlock,1,1), grid=(xBlocksPerGrid,1), stream=self.stream1)
            end_iter = timer()

            if debug:
                print('dev parents:', self.dev_parent)
                print('dev cost: ', self.dev_cost)
                print('dev unexplored: ', self.dev_unexplored)
                print('dev open: ', self.dev_open)
                print('dev threshold: ', self.dev_threshold)

                print('goal reached: ', goal_reached)
                print('y size: ', ySize, 'y: ' , dev_y)
                print('G size: ', gSize, 'G: ', dev_G)

                print('x size: ', dev_xSize, 'x: ', dev_x)
                print('######### iteration: ', iteration, end_iter-start_iter)











class GMTmemStream(object):
    def __init__(self, init_parameters, debug=False):
        self.route = []
        self.start = 0
        self.stream1 = drv.Stream()
        self.stream2 = drv.Stream()

        self._cpu_init(init_parameters, debug)
        self._gpu_init(debug)


    def _cpu_init(self, init_parameters, debug):
        self.states = init_parameters['states']
        self.n = self.states.shape[0]
        self.waypoints = np.arange(self.n).astype(np.int32)

        self.neighbors = init_parameters['neighbors']
        self.num_neighbors = init_parameters['num_neighbors']

        self.cost = np.full(self.n, np.inf).astype(np.float32)
        self.Vunexplored = np.full(self.n, 1).astype(np.int32)
        self.Vopen = np.zeros_like(self.Vunexplored).astype(np.int32)

        self.threadsPerBlock = init_parameters['threadsPerBlock']
        self.nBlocksPerGrid = int(((self.n + self.threadsPerBlock - 1) / self.threadsPerBlock))
        
        if debug:
            print('neighbors: ', self.neighbors)
            print('number neighbors: ', self.num_neighbors)

    def _gpu_init(self, debug):
        self.dev_n = cuda.to_gpu_async(np.array([self.n]).astype(np.int32), stream=self.stream2)

        self.neighbors_index = cuda.to_gpu_async(self.num_neighbors, stream=self.stream1)
        exclusiveScan(self.neighbors_index, stream=self.stream1)

        self.dev_num_neighbors = cuda.to_gpu_async(self.num_neighbors, stream=self.stream2)

        self.dev_states = cuda.to_gpu_async(self.states, stream=self.stream2)
        self.dev_waypoints = cuda.to_gpu_async(self.waypoints, stream=self.stream2)

        self.dev_neighbors = cuda.to_gpu_async(self.neighbors, stream=self.stream2)

        self.dev_Gindicator = cuda.GPUArray(self.Vopen.shape,self.Vopen.dtype)
        self.dev_xindicator = cuda.GPUArray(self.Vopen.shape,self.Vopen.dtype)

        self.dev_xindicator_zeros = cuda.GPUArray(self.Vopen.shape,self.Vopen.dtype)

        self.zero_val = np.zeros((), np.int32)

        self.dev_xindicator_zeros.fill(self.zero_val, stream = self.stream1)
        

        # self.stream1.synchronize()
        self.stream2.synchronize()


    def step_init(self, iter_parameters, debug):
        self.cost[self.start] = np.inf
        self.Vunexplored[self.start] = 1
        self.Vopen[self.start] = 0

        if self.start != iter_parameters['start'] and len(self.route) > 2:
            del self.route[-1]

        self.obstacles = iter_parameters['obstacles']
        self.num_obs = iter_parameters['num_obs']
        self.parent = np.full(self.n, -1).astype(np.int32)

        self.start = iter_parameters['start']
        self.goal = iter_parameters['goal']
        self.radius = iter_parameters['radius']
        self.threshold = np.array([ iter_parameters['threshold'] ]).astype(np.float32)

        self.cost[self.start] = 0
        self.Vunexplored[self.start] = 0
        self.Vopen[self.start] = 1

        if debug:
            print('parents:', self.parent)
            print('cost: ', self.cost)
            print('Vunexplored: ', self.Vunexplored)
            print('Vopen: ', self.Vopen)

        self.dev_open = cuda.to_gpu_async(self.Vopen, stream=self.stream2)

        self.dev_threshold = cuda.to_gpu_async(self.threshold, stream=self.stream1)

        self.dev_radius = cuda.to_gpu_async(np.array([self.radius]).astype(np.float32), stream=self.stream2)
        self.dev_obstacles = cuda.to_gpu_async(self.obstacles, stream=self.stream2) 
        self.dev_num_obs = cuda.to_gpu_async(self.num_obs, stream=self.stream2)

        self.dev_parent = cuda.to_gpu_async(self.parent, stream=self.stream2)
        self.dev_cost = cuda.to_gpu_async(self.cost, stream=self.stream1)
        
        self.dev_unexplored = cuda.to_gpu_async(self.Vunexplored, stream=self.stream1)


    def get_path(self):
        p = self.goal
        while p != -1:
            self.route.append(p)
            p = self.parent[p]

    def run_step(self, iter_parameters, iter_limit=1000, debug=False):
        self.step_init(iter_parameters,debug)

        goal_reached = False
        iteration = 0
        while True:
            iteration += 1

            start_iter = timer()
            ########## create Wave front ###############

            #start_wave_f= timer()
            wavefront(self.dev_Gindicator, self.dev_open, self.dev_cost, self.dev_threshold, self.dev_n, block=(self.threadsPerBlock,1,1), grid=(self.nBlocksPerGrid,1), stream=self.stream1)
            self.dev_threshold += 2* self.dev_radius
            goal_reached = self.dev_Gindicator[self.goal].get() == 1
            
            dev_Gscan = cuda.to_gpu_async(self.dev_Gindicator, stream=self.stream1)
            exclusiveScan(dev_Gscan, stream=self.stream1)
            

            dev_gSize = dev_Gscan[-1] + self.dev_Gindicator[-1]
            gSize = int(dev_gSize.get_async(stream=self.stream1))

            if iteration >= iter_limit:
                print('### iteration limit ###', iteration)
                return self.route
            elif goal_reached:
                print('### goal reached ### ', iteration)
                self.parent = self.dev_parent.get_async(stream=self.stream1)
                self.route = []
                self.get_path()
                return self.route
            elif gSize == 0:
                print('### threshold skip ', iteration)
                continue

            #end_wave_f= timer()

            dev_G = cuda.GPUArray([gSize,],np.int32)
            #dev_G = cuda.zeros(gSize, dtype=np.int32)
            
            compact(dev_G, dev_Gscan, self.dev_Gindicator, self.dev_waypoints, self.dev_n, block=(self.threadsPerBlock,1,1), grid=(self.nBlocksPerGrid,1), stream=self.stream1)    

            #end_wave_f= timer()

            ########## creating neighbors of wave front to connect open ###############
            
            #dev_xindicator = cuda.zeros_like(self.dev_open, dtype= np.int32,stream= self.stream1)

            #self.dev_xindicator.fill(self.zero_val, stream = self.stream1)
            #print(self.dev_xindicator_zeros.nbytes)
            
            drv.memcpy_dtod_async(self.dev_xindicator.gpudata,self.dev_xindicator_zeros.gpudata,self.dev_xindicator_zeros.nbytes,stream=self.stream1)

            
            gBlocksPerGrid = int(((gSize + self.threadsPerBlock - 1) / self.threadsPerBlock))
            
            neighborIndicator(self.dev_xindicator, dev_G, self.dev_unexplored, self.dev_neighbors, self.dev_num_neighbors, self.neighbors_index, dev_gSize, block=(self.threadsPerBlock,1,1), grid=(gBlocksPerGrid,1), stream=self.stream1)
            # start_create_n= timer()
            dev_xscan = cuda.to_gpu_async(self.dev_xindicator, stream=self.stream1)

            exclusiveScan(dev_xscan, stream=self.stream1)
            
            # dev_xSize = dev_xscan[-1] + self.dev_xindicator[-1]
            # end_create_n= timer()


            #start_create_n= timer()
            #dev_xscan = cuda.to_gpu_async(self.dev_xindicator, stream=self.stream1)
            #start_create_n= timer()
            #dev_xSize = cuda.sum(self.dev_xindicator, stream = self.stream1)
            #exclusiveScan(dev_xscan, stream=self.stream1)
            #start_create_n= timer()
            dev_xSize = dev_xscan[-1] + self.dev_xindicator[-1]
            #end_create_n= timer()

            xSize = int(dev_xSize.get_async(stream=self.stream1))
            
            if xSize == 0:
                print('### x skip')
                continue

            dev_x = cuda.GPUArray([xSize,],np.int32)
            #dev_x = cuda.zeros(xSize, dtype=np.int32)
            compact(dev_x, dev_xscan, self.dev_xindicator, self.dev_waypoints, self.dev_n, block=(self.threadsPerBlock,1,1), grid=(self.nBlocksPerGrid,1), stream=self.stream1)

            
            ######### scan and compact open set to connect neighbors ###############

            #start_scan_c= timer() 
            dev_yscan = cuda.to_gpu_async(self.dev_open, stream=self.stream2)
            exclusiveScan(dev_yscan, stream=self.stream2)
            dev_ySize = dev_yscan[-1] + self.dev_open[-1]
            ySize = int(dev_ySize.get_async(stream=self.stream2))

            #dev_y = cuda.zeros(ySize, dtype=np.int32)
            dev_y = cuda.GPUArray([ySize,],np.int32)
            compact(dev_y, dev_yscan, self.dev_open, self.dev_waypoints, self.dev_n, block=(self.threadsPerBlock,1,1), grid=(self.nBlocksPerGrid,1), stream=self.stream2)

            self.stream1.synchronize()
            self.stream2.synchronize()
            #end_scan_c= timer() 
            ######### connect neighbors ####################
            # # launch planning
            #start_planning= timer() 
            xBlocksPerGrid = int(((xSize + self.threadsPerBlock - 1) / self.threadsPerBlock))
            dubinConnection(self.dev_cost, self.dev_parent, dev_x, dev_y, self.dev_states, self.dev_open, self.dev_unexplored, dev_xSize, dev_ySize, self.dev_obstacles, self.dev_num_obs, self.dev_radius, block=(self.threadsPerBlock,1,1), grid=(xBlocksPerGrid,1), stream=self.stream1)
            #end_planning= timer()
            end_iter = timer()
            if debug:
                print('dev parents:', self.dev_parent)
                print('dev cost: ', self.dev_cost)
                print('dev unexplored: ', self.dev_unexplored)
                print('dev open: ', self.dev_open)
                print('dev threshold: ', self.dev_threshold)

                print('goal reached: ', goal_reached)
                print('y size: ', ySize, 'y: ' , dev_y)
                print('G size: ', gSize, 'G: ', dev_G)

                print('x size: ', dev_xSize, 'x: ', dev_x)
                print('######### iteration: ', iteration, 'Time Taken: ', end_iter-start_iter)
