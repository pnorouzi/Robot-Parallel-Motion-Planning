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
    __global__ void neighborIndicator(int *x_indicator, int *G, int *Vunexplored, int *neighbors, int *num_neighbors, int *neighbors_index, const int n){
        const int index = threadIdx.x;
        if(index > n){
            return;
        }

        for(int i=0; i < num_neighbors[index]; i++){
            int j = neighbors[G[index]*neighbors_index[index] + i];
            x_indicator[j] = Vunexplored[j] || x_indicator[j] > 0 ? 1 : 0;
        }      
    }

    __global__ void wavefront(int *G, int *Vopen, float *cost, float threshold, const int n){
        const int index = threadIdx.x;
        if(index > n){
            return;
        }
        
        G[index] =  Vopen[index] && cost[index] <= threshold ? 1 : 0;
    }

    __global__ void compact(int *x, int *x_scan, int *x_indicator, int *waypoints, const int n){
        const int index = threadIdx.x;
        if(index > n){
            return;
        }

        if(x_indicator[index] == 1){
            x[x_scan[index]] = waypoints[index];
        }
    }

    __device__ bool dubinCost(float *cost, float *x, float *y){
        float new_cost = powf(x[0]-y[0],2) + powf(x[1]-y[1],2) + powf(x[2]-y[2],2);
        bool connected = new_cost < *cost;
        *cost = connected ? new_cost : *cost;
        return connected;

    }

    __global__ void dubinConnection(float *cost, int *parent, int *x, int *y, float *states, int *Vopen, int *Vunexplored, const int xSize, const int ySize){
        const int index = threadIdx.x;
        if(index > xSize){
            return;
        }

        for(int i=0; i < ySize; i++){
            bool connected = dubinCost(&cost[x[index]], &states[x[index]*3], &states[y[i]*3]);
            parent[x[index]] = connected ? y[i] : parent[x[index]];
            Vopen[x[index]] = !connected;
            Vopen[y[i]] = connected;
            Vunexplored[x[index]] = connected;
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
    threshold = np.array([1]).astype(np.float32)

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
    ########## Create Wave front ###############
    dev_Gindicator = cuda.zeros_like(dev_open, dtype=np.int32)
    wavefront(dev_Gindicator, dev_open, dev_cost, dev_threshold, dev_n, block=(n,1,1), grid=(1,1))

    dev_Gscan = cuda.zeros_like(dev_Gindicator, dtype=np.int32)
    exclusiveScan(dev_Gscan)
    dev_gSize = dev_Gscan[-1]
    gSize = int(dev_gSize.get())

    dev_G = cuda.zeros(dev_gSize, dtype=np.int32)
    compact(dev_G, dev_Gscan, dev_Gindicator, dev_waypoints, dev_gSize, block=(n,1,1), grid=(1,1))
    print('G: ', dev_G.get())

    ########## creating neighbors of wave front to connect open ###############
    dev_xindicator = cuda.zeros_like(dev_open, dtype=np.int32)
    neighborIndicator(dev_xindicator, dev_G, dev_unexplored, dev_neighbors, dev_num_neighbors, neighbors_index, dev_gSize, block=(gSize,1,1), grid=(1,1))


    dev_xscan = cuda.zeros_like(dev_xindicator, dtype=np.int32)
    exclusiveScan(dev_xscan)
    dev_xSize = dev_xscan[-1]
    xSize = int(dev_xSize.get())

    dev_x = cuda.zeros(dev_xSize, dtype=np.int32)
    compact(dev_x, dev_xscan, dev_xindicator, dev_waypoints, dev_xSize, block=(n,1,1), grid=(1,1))
    print('x: ', dev_x.get())

    ######### scan and compact open set to connect neighbors ###############
    dev_yscan = dev_open
    exclusiveScan(dev_yscan)
    dev_ySize = dev_yscan[-1]
    ySize = int(dev_ySize.get())

    dev_y = cuda.zeros(dev_ySize, dtype=np.int32)
    compact(dev_y, dev_yscan, dev_open, dev_waypoints, dev_ySize, block=(n,1,1), grid=(1,1))
    print('y: ', dev_y.get())

    ######### connect neighbors ####################
    # # launch planning
    dubinConnection(dev_cost, dev_parent, dev_x, dev_y, dev_states, dev_open, dev_unexplored, dev_xSize, dev_ySize, block=(xSize,1,1), grid=(1,1))
    print('################ post GMT* ###############')
    print('parents:', dev_parent.get())
    print('cost: ', dev_cost.get())
    print('Vunexplored: ', dev_unexplored.get())
    print('Vopen: ', dev_open.get())