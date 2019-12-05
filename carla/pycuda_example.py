import pycuda
from pycuda.scan import ExclusiveScanKernel
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as cuda
from pycuda.compiler import SourceModule

import numpy as np
import math

''''

'''

from gmt_planner import *


if __name__ == '__main__':
    states = np.array([[10,2,135*np.pi/180], [10,2,90*np.pi/180], [10,2,45*np.pi/180], # 0-2
        [8,5,135*np.pi/180], [8,5,90*np.pi/180], [8,5,45*np.pi/180], # 3-5
        [12,6,135*np.pi/180], [12,6,90*np.pi/180], [12,6,45*np.pi/180], # 6-8
        [11,8,135*np.pi/180], [11,8,90*np.pi/180], [11,8,45*np.pi/180], # 9-11
        [2,7,135*np.pi/180], [2,7,90*np.pi/180], [2,7,45*np.pi/180], # 12-14
        [5,10,135*np.pi/180], [5,10,90*np.pi/180], [5,10,45*np.pi/180]]).astype(np.float32) #15-17

    n0 = [3,4,5,6,7,8]
    n1 = [0,1,2,9,10,11,12,13,14,15,16,17]
    n2 = [3,4,5,9,10,11]
    n3 = [3,4,5,6,7,8,15,16,17]
    n4 = [3,4,5,15,16,17]
    n5 = [3,4,5,9,10,11,12,13,14]
    nn = 3*n0 + 3*n1 + 3*n2 + 3*n3 + 3*n4 + 3*n5

    neighbors = np.array(nn).astype(np.int32)
    num_neighbors = np.array([6,6,6, 12,12,12, 6,6,6, 9,9,9, 6,6,6, 9,9,9]).astype(np.int32)

    obstacles = np.array([[7,6,4,9]]).astype(np.float32)
    num_obs = np.array([1]).astype(np.int32)

    start = 1
    goal = 15
    radius = 2
    threshold = 10

    init_parameters = {'states':states, 'neighbors':neighbors, 'num_neighbors':num_neighbors}
    iter_parameters = {'start':start, 'goal':goal, 'radius':radius, 'threshold':threshold, 'obstacles':obstacles, 'num_obs':num_obs}

    gmt = GMT(init_parameters, debug=True)
    route = gmt.run_step(iter_parameters, iter_limit=8, debug=True)
    print(route)