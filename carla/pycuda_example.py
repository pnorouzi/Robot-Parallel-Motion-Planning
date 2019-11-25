import math

import numpy as np
import networkx as nx

import carla

import pycuda.autoinit
import pycuda.driver as drv

from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] * b[i];
}

__global__ void np_test(float *dest, float *c, int n){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i > n){
        return;
    }
    dest[i] = c[i];
}


""")




if __name__ == '__main__':

    multiply_them = mod.get_function("multiply_them")
    np_test = mod.get_function("np_test")

    a = np.random.randn(400).astype(np.float32)
    b = np.random.randn(400).astype(np.float32)

    c = np.array([np.array([1,2,3]), np.array([1,2]), np.array([4])])
    n = np.array([6])

    dest = np.zeros_like(c)
    print(dest)
    # multiply_them(drv.Out(dest), drv.In(a), drv.In(b), block=(400,1,1), grid=(1,1))

    np_test(drv.Out(dest), drv.In(c), drv.In(n), block=(6,1,1), grid=(1,1))    

    print(dest)
    print(c)

    start = 0

    n = 6
    parent = np.full((n,1), -1)
    cost = np.full((n,1), np.inf)
    cost[start] = 0

    Vunexplored = np.full((n,1), 1) 
    Vopen = np.full((n,1), 0)

    Vopen[start] = 1

    # unexplored[start] = 0
