import math

import numpy as np
import networkx as nx

import carla

import pycuda
from pycuda.scan import ExclusiveScanKernel
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

# from codepy.cgen import *
# from codepy.bpl import BoostPythonModule
# from codepy.cuda import CudaModule


mod = SourceModule("""
    __global__ void neighborIndicator(int *x_indicator, int *G, int *Vunexplored, int *neighbors, int *num_neighbors, int *neighbors_index, const int n){
        const int index = threadIdx.x;
        if(index > n){
            return;
        }
        if(!G[index]){
            return;
        }
        for(int i=0; i < num_neighbors[index]; i++){
            int j = neighbors[index*neighbors_index[index] + i];
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
            Vopen[x[index]] = connected;
            Vunexplored[x[index]] = !connected;
        }
    }
""")




if __name__ == '__main__':

    start = 0
    goal = 5
    num = 6
    n = np.array([num])
    threshold = np.array([1]).astype(np.float32)

    wavefront = mod.get_function("wavefront")
    neighborIndicator = mod.get_function("neighborIndicator")
    exclusiveScan = ExclusiveScanKernel(np.int32, "a+b", 0)
    compact = mod.get_function("compact")
    dubinConnection = mod.get_function("dubinConnection")

    ####### INIT #########
    states = np.array([[1,2,1], [0,3,3], [0,3,4], [1,2,0], [1,2,2], [0,3,2]]).astype(np.float32)
    waypoints = np.array([0,1,2,3,4,5]).astype(np.int32)

    neighbors = np.array([1,2, 0,3, 0,3,4, 1,2,4, 2,3,5, 4]).astype(np.int32)
    print('neighbors: ',neighbors)

    num_neighbors = np.array([2, 2, 3, 3, 3, 1]).astype(np.int32)
    neighbors_index = gpuarray.to_gpu(num_neighbors)
    exclusiveScan(neighbors_index)

    parent = np.full(num, -1)
    print('parents:', parent)

    cost = np.full(num, np.inf).astype(np.float32)
    cost[start] = 0
    print('cost: ', cost)

    Vunexplored = np.full(num, 1).astype(np.int32)
    Vunexplored[start] = 0
    print('Vunexplored: ', Vunexplored)

    Vopen = np.zeros_like(Vunexplored).astype(np.int32)
    Vopen[start] = 1
    print('Vopen: ', Vopen)

    ########## algorithm starts ####################
    ########## Create Wave front ###############
    G = np.zeros_like(Vopen).astype(np.int32)
    wavefront(drv.InOut(G), drv.In(Vopen), drv.In(cost), drv.In(threshold), drv.In(n), block=(num,1,1), grid=(1,1))
    print('G: ', G)

    ########## creating neighbors of wave front to connect open ###############
    x_indicator = np.zeros_like(waypoints).astype(np.int32)
    neighborIndicator(drv.InOut(x_indicator), drv.In(G), drv.In(Vunexplored), drv.In(neighbors), drv.In(num_neighbors), neighbors_index, drv.In(n), block=(num,1,1), grid=(1,1))

    print('x_indicator: ', x_indicator)

    ######## scan and compact neighbor set ##################
    x_scan = gpuarray.to_gpu(x_indicator)
    exclusiveScan(x_scan)
    x_scan_cpu = x_scan.get()
    print('x_scan_cpu: ', x_scan_cpu)
    xSize = int(x_scan_cpu[-1])

    np_xSize = np.array([xSize]).astype(np.int32)
    print('np_xSize: ', np_xSize)

    x = np.zeros(xSize).astype(np.int32)
    compact(drv.InOut(x), x_scan, drv.In(x_indicator), drv.In(waypoints), drv.In(np_xSize), block=(num,1,1), grid=(1,1))
    print('x: ', x)

    ######### scan and compact open set to connect neighbors ###############
    y_scan = gpuarray.to_gpu(Vopen)
    exclusiveScan(y_scan)
    y_scan_cpu = y_scan.get()
    print('y_scan_cpu: ', y_scan_cpu)
    ySize = int(y_scan_cpu[-1])

    np_ySize = np.array([ySize]).astype(np.int32)
    print('np_ySize: ', np_ySize)

    y = np.zeros(ySize).astype(np.int32)
    compact(drv.InOut(y), y_scan, drv.In(Vopen), drv.In(waypoints), drv.In(np_ySize), block=(num,1,1), grid=(1,1))
    print('y: ', y)


    ######### connect neighbors ####################
    # # launch planning
    dubinConnection(drv.InOut(cost), drv.InOut(parent), drv.In(x), drv.In(y), drv.In(states), drv.InOut(Vopen), drv.InOut(Vunexplored), drv.In(np_xSize), drv.In(np_ySize), block=(xSize,1,1), grid=(1,1))
    print(cost)
