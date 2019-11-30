import pycuda
from pycuda.scan import ExclusiveScanKernel
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as cuda
from pycuda.compiler import SourceModule

import numpy as np

''''

'''

mod = SourceModule("""
    __global__ void wavefront(int *G, int *open, float *cost, float threshold, const int n){
        const int index = threadIdx.x;
        if(index > n){
            return;
        }
        
        G[index] =  open[index] && cost[index] <= threshold ? 1 : 0;
    }

    __global__ void neighborIndicator(int *x_indicator, int *G, int *unexplored, int *neighbors, int *num_neighbors, int *neighbors_index, const int n){
        const int index = threadIdx.x;
        if(index > n){
            return;
        }

        for(int i=0; i < num_neighbors[G[index]]; i++){
            int j = neighbors[neighbors_index[G[index]] + i];
            x_indicator[j] = unexplored[j] || x_indicator[j] > 0 ? 1 : 0;
        }      
    }

    __device__ bool dubinCost(float *cost, float *x, float *y){
        float new_cost = powf(x[0]-y[0],2) + powf(x[1]-y[1],2) + powf(x[2]-y[2],2);
        bool connected = new_cost < *cost;
        *cost = connected ? new_cost : *cost;
        return connected;

    }

    __global__ void dubinConnection(float *cost, int *parent, int *x, int *y, float *states, int *open, int *unexplored, const int xSize, const int ySize){
        const int index = threadIdx.x;
        if(index > xSize){
            return;
        }

        for(int i=0; i < ySize; i++){
            bool connected = dubinCost(&cost[x[index]], &states[x[index]*3], &states[y[i]*3]);
            parent[x[index]] = connected ? y[i] : parent[x[index]];
            cost[x[index]] = connected ? cost[y[i]] + cost[x[index]] : cost[x[index]];
            open[x[index]] = connected ? 1 : open[x[index]];
            open[y[i]] = connected ? 0 : open[y[i]];
            unexplored[x[index]] = connected ? 1 : unexplored[x[index]];
        }
    }

    __global__ void compact(int *x, int *scan, int *indicator, int *waypoints, const int n){
        const int index = threadIdx.x;
        if(index > n){
            return;
        }

        if(indicator[index] == 1){
            x[scan[index]] = waypoints[index];
        }
    }
""")

if __name__ == '__main__':

    wavefront = mod.get_function("wavefront")
    neighborIndicator = mod.get_function("neighborIndicator")
    exclusiveScan = ExclusiveScanKernel(np.int32, "a+b", 0)
    compact = mod.get_function("compact")
    dubinConnection = mod.get_function("dubinConnection")

    ### CPU INIT ###
    print('################ CPU INIT ###############')
    states = np.array([[1,2,1], [0,3,3], [0,3,4], [1,2,0], [1,2,2], [0,3,2]]).astype(np.float32)
    waypoints = np.array([0,1,2,3,4,5]).astype(np.int32)

    start = 0
    goal = 5
    n = 6
    radius = 6
    threshold = np.array([radius]).astype(np.float32)

    neighbors = np.array([1,2, 0,3, 0,3,4, 1,2,4, 2,3,5, 4]).astype(np.int32)
    print('neighbors: ',neighbors)
    num_neighbors = np.array([2, 2, 3, 3, 3, 1]).astype(np.int32)

    parent = np.full(n, -1).astype(np.int32)
    print('parents:', parent)

    cost = np.full(n, np.inf).astype(np.float32)
    cost[start] = 0
    print('cost: ', cost)

    Vunexplored = np.full(n, 1).astype(np.int32)
    Vunexplored[start] = 0
    print('Vunexplored: ', Vunexplored)

    Vopen = np.zeros_like(Vunexplored).astype(np.int32)
    Vopen[start] = 1
    print('Vopen: ', Vopen)

    ### GPU INIT ###
    print('################ GPU INIT ###############')
    dev_states = cuda.to_gpu(states)
    dev_waypoints = cuda.to_gpu(waypoints)

    dev_threshold = cuda.to_gpu(threshold)
    dev_n = cuda.to_gpu(np.array([n]))

    dev_neighbors = cuda.to_gpu(neighbors)
    dev_num_neighbors = cuda.to_gpu(num_neighbors)
    neighbors_index = cuda.to_gpu(num_neighbors)
    exclusiveScan(neighbors_index)

    dev_parent = cuda.to_gpu(parent)
    
    dev_cost = cuda.to_gpu(cost)

    dev_open = cuda.to_gpu(Vopen)
    dev_unexplored = cuda.to_gpu(Vunexplored)

    print('################ GMT* ###############')
    ########## algorithm starts ####################
    i = 0
    while True:
        print('######### iteration: ', i)
        print('parents:', dev_parent)
        print('cost: ', dev_cost)
        print('Vunexplored: ', dev_unexplored)
        print('Vopen: ', dev_open)
        print('threshold: ', dev_threshold)
        i += 1

        ########## create Wave front ###############
        dev_Gindicator = cuda.zeros_like(dev_open, dtype=np.int32)
        wavefront(dev_Gindicator, dev_open, dev_cost, dev_threshold, dev_n, block=(n,1,1), grid=(1,1))
        dev_threshold += dev_threshold
        goal_reached = dev_Gindicator[goal].get() == 1

        ######### scan and compact open set to connect neighbors ###############
        dev_yscan = cuda.to_gpu(dev_open)
        exclusiveScan(dev_yscan)
        dev_ySize = dev_yscan[-1]
        ySize = int(dev_ySize.get())

        dev_y = cuda.zeros(ySize, dtype=np.int32)
        compact(dev_y, dev_yscan, dev_open, dev_waypoints, dev_ySize, block=(n,1,1), grid=(1,1))
        
        dev_Gscan = cuda.to_gpu(dev_Gindicator)
        exclusiveScan(dev_Gscan)
        dev_gSize = dev_Gscan[-1]
        gSize = int(dev_gSize.get())

        print('goal reached: ', goal_reached)
        print('y size: ', ySize, 'y: ' , dev_y)
        print('G size: ', gSize, 'G: ', dev_Gindicator)

        if ySize == 0 or i == 5:
            print('### empty open set ###')
            break
        elif goal_reached:
            print('### goal reached ###')
            break
        elif gSize == 0:
            print('### threshold skip')
            continue

        dev_G = cuda.zeros(gSize, dtype=np.int32)
        compact(dev_G, dev_Gscan, dev_Gindicator, dev_waypoints, dev_gSize, block=(n,1,1), grid=(1,1))
        

        ########## creating neighbors of wave front to connect open ###############
        dev_xindicator = cuda.zeros_like(dev_open, dtype=np.int32)
        neighborIndicator(dev_xindicator, dev_G, dev_unexplored, dev_neighbors, dev_num_neighbors, neighbors_index, dev_gSize, block=(gSize,1,1), grid=(1,1))

        dev_xscan = cuda.to_gpu(dev_xindicator)
        exclusiveScan(dev_xscan)
        dev_xSize = dev_xscan[-1]
        xSize = int(dev_xSize.get())

        if xSize == 0:
            print('### x skip')
            continue

        dev_x = cuda.zeros(xSize, dtype=np.int32)
        compact(dev_x, dev_xscan, dev_xindicator, dev_waypoints, dev_xSize, block=(n,1,1), grid=(1,1))
        print('x size: ', xSize, 'x: ', dev_x)

        ######### connect neighbors ####################
        # # launch planning
        dubinConnection(dev_cost, dev_parent, dev_x, dev_y, dev_states, dev_open, dev_unexplored, dev_xSize, dev_ySize, block=(xSize,1,1), grid=(1,1))