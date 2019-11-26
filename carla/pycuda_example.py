import math

import numpy as np
import networkx as nx

import carla

import pycuda
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from codepy.cgen import *
from codepy.bpl import BoostPythonModule
from codepy.cuda import CudaModule

import numpy as np


mod = SourceModule("""
__global__ void np_test(float *dest, float *c, int n){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i > n){
        return;
    }
    dest[i] = c[i];
}

__global__ void wavefront(float *G, float *Vopen, float *cost, float *threshold, int n){
    const int idx = threadIdx.x;
    if(idx > n){
        return;
    }

    G[idx] = cost[idx] < threshold ? 1 : 0;
}

__global__ void neighborIndicator(float *x_indicator, float *G, float *Vunexplored, float *neighbors, int n){
    const int idx = threadIdx.x;
    if(idx > n){
        return;
    }

    for(int i=0; i < num_neighbors[idx]; i++){
        x_indicator[neighbors[idx][i]] = Vunexplored[idx] ? 1 : 0;
    }   
}

__global__ void dubinConnection(float *cost, int *partent, float *x, float *Vopen, int n, int xSize){
    const int idx = threadIdx.x;
    if(idx > xSize){
        return;
    }

    for(int i=0; i < n; i++){
        dubinCost(cost[idx], parent[idx], x[idx], Vopen[i]);
    }
}

__device__ void dubinCost(float *cost, int *parent, float *x, float *Vopen){
    
}


""")




if __name__ == '__main__':

    np_test = mod.get_function("np_test")
    wavefront = mod.get_function("wavefront")
    neighborIndicator = mod.get_function("neighborIndicator")
    dubinConnection = mod.get_function("dubinConnection")

    start = 0
    goal = 5
    n = np.array([6])
    threshold = 0
    
    neighbors = np.array([np.array([1,2]), np.array([0,3]), np.array([0,3,4]), np.array([1,2,4]), np.array([2,3,5]), np.array([4])])
    num_neighbors = np.array([2, 2, 3, 3, 3, 1])

    parent = np.full((n,1), -1)

    cost = np.full((n,1), np.inf)
    cost[start] = 0

    Vunexplored = np.full((n,1), 1)
    Vunexplored[start] = 0

    Vopen = np.zeros_like(Vunexplored)
    Vopen[start] = 1

    G = np.zeros_like(Vopen)
    wavefront(drv.Out(G), drv.In(Vopen), drv.In(cost), threshold, n, block=(6,1,1), grid=(1,1))

    # run kernel

    x_indicator = np.zeros_like(G)
    neighborIndicator(drv.Out(x_indicator), drv.In(G), drv.In(Vunexplored), drv.In(neighbors), drv.In(num_neighbors), n, block=(6,1,1), grid=(1,1))

    # xSize = thrust::compact

    x = np.zeros([xSize, 1])
    # populate x with indices

    # launch planning
    dubinConnection(drv.Out(cost), drv.Out(parent), drv.In(x), drv.In(Vopen), n, xSize)

