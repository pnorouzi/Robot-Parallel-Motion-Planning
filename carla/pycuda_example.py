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
    __device__ bool check_col(float *y_vals,float *x_vals,float *obstacles, int num_obs){

        for (int obs=0;obs<num_obs;obs++){
            for (int i=0;i<150;i++){
                if ((obstacles[obs*4 +3]<=y_vals[i] || obstacles[obs*4 +1]>=y_vals[i]) && (obstacles[obs*4]<=x_vals[i] || obstacles[obs*4 + 2]>=x_vals[i])){
                return false;
                }
            }
        }
        return false;
    }

    __device__ void RSRcost(float *curCost, float *point1,float *point2, int r_min, float *obstacles, int num_obs){
        float PI = 3.1415926535;

        float p_c1 [2] = { point1[0] + (r_min * cosf(point1[2] - PI/2)), point1[1] + (r_min * sinf(point1[2] - PI/2))}; 
        float p_c2 [2] = { point2[0] + (r_min * cosf(point2[2] - PI/2)), point2[1] + (r_min * sinf(point2[2] - PI/2))};

        float r_1 = sqrtf(powf(p_c1[0]-point1[0],2.0) + powf(p_c1[1]-point1[1],2.0));
        float r_2 = sqrtf(powf(p_c2[0]-point2[0],2.0) + powf(p_c2[1]-point2[1],2.0));

        float V1 [2] = {p_c2[0]-p_c1[0],p_c2[1]-p_c1[1]};

        float dist_centers = sqrtf(powf(V1[0],2) + powf(V1[1],2));

        float c = (r_1-r_2)/dist_centers;
        V1[0] /= dist_centers;
        V1[1] /= dist_centers;

        float normal [2] = {(V1[0]*c)-(V1[1]*sqrtf(1-powf(c,2))),(V1[0]*sqrtf(1-powf(c,2)))+(V1[1]*c)};

        if (isnan(normal[0])){
            return;
        }

        float tangent_1 [2] = {p_c1[0] + (r_1* normal[0]),p_c1[1] + (r_1* normal[1])};
        float tangent_2 [2] = {p_c2[0] + (r_2* normal[0]),p_c2[1] + (r_2* normal[1])};

        float V2 [2] = {tangent_2[0]-tangent_1[0],tangent_2[1]-tangent_1[1]};


        float p2_h [2] = {point1[0], point1[1]};
        float v1 [2] = {p2_h[0]-p_c1[0], p2_h[1]-p_c1[1]};
        float v2 [2] = {tangent_1[0]-p_c1[0], tangent_1[1]-p_c1[1]};

        float theta_1 = atan2f(v2[1],v2[0]) - atan2f(v1[1],v1[0]);

        if (theta_1>0){
            theta_1-=(PI*2);
        }

        float angle = point1[2] + (PI/2);

        float x_vals [150] = { };
        float y_vals [150] = { };
        float d_theta = theta_1/50;

        for (int i=0;i<50;i++){
            x_vals[i] = (abs(r_1) * cosf(angle+(i*d_theta))) + p_c1[0];
            y_vals[i] = (abs(r_1) * sinf(angle+(i*d_theta))) + p_c1[1];
        }


        float p3_h [2] = {point2[0], point2[1]};
        v1[0] = tangent_2[0]-p_c2[0];
        v1[1] = tangent_2[1]-p_c2[1];

        v2[0] = p3_h[0] - p_c2[0];
        v2[1] = p3_h[1] - p_c2[1];

        float theta_2 = atan2f(v2[1],v2[0]) - atan2f(v1[1],v1[0]);

        if (theta_2>0){
            theta_2-=(PI*2);
        }

        angle = atan2f((tangent_2[1]-p_c2[1]),(tangent_2[0]-p_c2[0]));

        d_theta = theta_2/50;

        for (int i=0;i<50;i++){
            x_vals[i+100] = (abs(r_2) * cosf(angle+(i*d_theta))) + p_c2[0];
            y_vals[i+100] = (abs(r_2) * sinf(angle+(i*d_theta))) + p_c2[1];
        }

        float d_x = (x_vals[99] - x_vals[49])/50;
        float d_y = (y_vals[99] - y_vals[49])/50;

        for (int i=0;i<50;i++){
            x_vals[i+50] = x_vals[49] + (i*d_x);
            y_vals[i+50] = y_vals[49] + (i*d_y);
        }

        // checks for collision

        bool collision = check_col(y_vals,x_vals,obstacles,num_obs);


        if (collision){
            return;
        }


        float cost = abs((r_1*theta_1)) + abs((r_2*theta_2)) + sqrtf(powf(V2[0],2) + powf(V2[1],2));

        if (cost> *curCost){
            return;
        }

        *curCost = cost;
        return;
    }

    __device__ void LSLcost(float *curCost, float *point1,float *point2, int r_min, float *obstacles, int num_obs){
        float PI = 3.1415926535;

        float p_c1 [2] = { point1[0] + (r_min * cosf(point1[2] + PI/2)), point1[1] + (r_min * sinf(point1[2] + PI/2))}; 
        float p_c2 [2] = { point2[0] + (r_min * cosf(point2[2] + PI/2)), point2[1] + (r_min * sinf(point2[2] + PI/2))};

        float r_1 = -1.0 * sqrtf(powf(p_c1[0]-point1[0],2.0) + powf(p_c1[1]-point1[1],2.0));
        float r_2 = -1.0 * sqrtf(powf(p_c2[0]-point2[0],2.0) + powf(p_c2[1]-point2[1],2.0));

        float V1 [2] = {p_c2[0]-p_c1[0],p_c2[1]-p_c1[1]};

        float dist_centers = sqrtf(powf(V1[0],2) + powf(V1[1],2));

        float c = (r_1-r_2)/dist_centers;
        V1[0] /= dist_centers;
        V1[1] /= dist_centers;

        float normal [2] = {(V1[0]*c)-(V1[1]*sqrtf(1-powf(c,2))),(V1[0]*sqrtf(1-powf(c,2)))+(V1[1]*c)};

        if (isnan(normal[0])){
            return;
        }

        float tangent_1 [2] = {p_c1[0] + (r_1* normal[0]),p_c1[1] + (r_1* normal[1])};
        float tangent_2 [2] = {p_c2[0] + (r_2* normal[0]),p_c2[1] + (r_2* normal[1])};

        float V2 [2] = {tangent_2[0]-tangent_1[0],tangent_2[1]-tangent_1[1]};


        float p2_h [2] = {point1[0], point1[1]};
        float v1 [2] = {p2_h[0]-p_c1[0], p2_h[1]-p_c1[1]};
        float v2 [2] = {tangent_1[0]-p_c1[0], tangent_1[1]-p_c1[1]};

        float theta_1 = atan2f(v2[1],v2[0]) - atan2f(v1[1],v1[0]);

        if (theta_1<0){
            theta_1+=(PI*2);
        }

        float angle = point1[2] - (PI/2);

        float x_vals [150] = { };
        float y_vals [150] = { };
        float d_theta = theta_1/50;

        for (int i=0;i<50;i++){
            x_vals[i] = (abs(r_1) * cosf(angle+(i*d_theta))) + p_c1[0];
            y_vals[i] = (abs(r_1) * sinf(angle+(i*d_theta))) + p_c1[1];
        }


        float p3_h [2] = {point2[0], point2[1]};
        v1[0] = tangent_2[0]-p_c2[0];
        v1[1] = tangent_2[1]-p_c2[1];

        v2[0] = p3_h[0] - p_c2[0];
        v2[1] = p3_h[1] - p_c2[1];

        float theta_2 = atan2f(v2[1],v2[0]) - atan2f(v1[1],v1[0]);

        if (theta_2<0){
            theta_2+=(PI*2);
        }

        angle = atan2f((tangent_2[1]-p_c2[1]),(tangent_2[0]-p_c2[0]));

        d_theta = theta_2/50;

        for (int i=0;i<50;i++){
            x_vals[i+100] = (abs(r_2) * cosf(angle+(i*d_theta))) + p_c2[0];
            y_vals[i+100] = (abs(r_2) * sinf(angle+(i*d_theta))) + p_c2[1];
        }

        float d_x = (x_vals[99] - x_vals[49])/50;
        float d_y = (y_vals[99] - y_vals[49])/50;

        for (int i=0;i<50;i++){
            x_vals[i+50] = x_vals[49] + (i*d_x);
            y_vals[i+50] = y_vals[49] + (i*d_y);
        }

        bool collision = check_col(y_vals,x_vals,obstacles,num_obs);

        if (collision){
            return;
        }


        float cost = abs((r_1*theta_1)) + abs((r_2*theta_2)) + sqrtf(powf(V2[0],2) + powf(V2[1],2));

        if (cost> *curCost){
            return;
        }

        *curCost = cost;
        return;
    }

    __device__ void LSRcost(float *curCost, float *point1,float *point2, int r_min, float *obstacles, int num_obs){
        float PI = 3.1415926535;

        float p_c1 [2] = { point1[0] + (r_min * cosf(point1[2] + PI/2)), point1[1] + (r_min * sinf(point1[2] + PI/2))}; 
        float p_c2 [2] = { point2[0] + (r_min * cosf(point2[2] - PI/2)), point2[1] + (r_min * sinf(point2[2] - PI/2))};

        float r_1 = -1.0 * sqrtf(powf(p_c1[0]-point1[0],2.0) + powf(p_c1[1]-point1[1],2.0));
        float r_2 = sqrtf(powf(p_c2[0]-point2[0],2.0) + powf(p_c2[1]-point2[1],2.0));

        float V1 [2] = {p_c2[0]-p_c1[0],p_c2[1]-p_c1[1]};

        float dist_centers = sqrtf(powf(V1[0],2) + powf(V1[1],2));

        float c = (r_1-r_2)/dist_centers;
        V1[0] /= dist_centers;
        V1[1] /= dist_centers;

        float normal [2] = {(V1[0]*c)-(V1[1]*sqrtf(1-powf(c,2))),(V1[0]*sqrtf(1-powf(c,2)))+(V1[1]*c)};

        if (isnan(normal[0])){
            return;
        }

        float tangent_1 [2] = {p_c1[0] + (r_1* normal[0]),p_c1[1] + (r_1* normal[1])};
        float tangent_2 [2] = {p_c2[0] + (r_2* normal[0]),p_c2[1] + (r_2* normal[1])};

        float V2 [2] = {tangent_2[0]-tangent_1[0],tangent_2[1]-tangent_1[1]};


        float p2_h [2] = {point1[0], point1[1]};
        float v1 [2] = {p2_h[0]-p_c1[0], p2_h[1]-p_c1[1]};
        float v2 [2] = {tangent_1[0]-p_c1[0], tangent_1[1]-p_c1[1]};

        float theta_1 = atan2f(v2[1],v2[0]) - atan2f(v1[1],v1[0]);

        if (theta_1<0){
            theta_1+=(PI*2);
        }

        float angle = point1[2] - (PI/2);

        float x_vals [150] = { };
        float y_vals [150] = { };
        float d_theta = theta_1/50;

        for (int i=0;i<50;i++){
            x_vals[i] = (abs(r_1) * cosf(angle+(i*d_theta))) + p_c1[0];
            y_vals[i] = (abs(r_1) * sinf(angle+(i*d_theta))) + p_c1[1];
        }

        float p3_h [2] = {point2[0], point2[1]};
        v1[0] = tangent_2[0]-p_c2[0];
        v1[1] = tangent_2[1]-p_c2[1];

        v2[0] = p3_h[0] - p_c2[0];
        v2[1] = p3_h[1] - p_c2[1];

        float theta_2 = atan2f(v2[1],v2[0]) - atan2f(v1[1],v1[0]);

        if (theta_2>0){
            theta_2-=(PI*2);
        }

        angle = atan2f((tangent_2[1]-p_c2[1]),(tangent_2[0]-p_c2[0]));

        d_theta = theta_2/50;

        for (int i=0;i<50;i++){
            x_vals[i+100] = (abs(r_2) * cosf(angle+(i*d_theta))) + p_c2[0];
            y_vals[i+100] = (abs(r_2) * sinf(angle+(i*d_theta))) + p_c2[1];
        }

        float d_x = (x_vals[99] - x_vals[49])/50;
        float d_y = (y_vals[99] - y_vals[49])/50;

        for (int i=0;i<50;i++){
            x_vals[i+50] = x_vals[49] + (i*d_x);
            y_vals[i+50] = y_vals[49] + (i*d_y);
        }

        bool collision = check_col(y_vals,x_vals,obstacles,num_obs);

        if (collision){
            return;
        }

        float cost = abs((r_1*theta_1)) + abs((r_2*theta_2)) + sqrtf(powf(V2[0],2) + powf(V2[1],2));

        if (cost> *curCost){
            return;
        }

        *curCost = cost;
        return;
    }

    __device__ void RSLcost(float *curCost, float *point1,float *point2, int r_min, float *obstacles, int num_obs){
        float PI = 3.1415926535;

        float p_c1 [2] = { point1[0] + (r_min * cosf(point1[2] - PI/2)), point1[1] + (r_min * sinf(point1[2] - PI/2))}; 
        float p_c2 [2] = { point2[0] + (r_min * cosf(point2[2] + PI/2)), point2[1] + (r_min * sinf(point2[2] + PI/2))};

        float r_1 = sqrtf(powf(p_c1[0]-point1[0],2.0) + powf(p_c1[1]-point1[1],2.0));
        float r_2 = -1.0 * sqrtf(powf(p_c2[0]-point2[0],2.0) + powf(p_c2[1]-point2[1],2.0));

        float V1 [2] = {p_c2[0]-p_c1[0],p_c2[1]-p_c1[1]};

        float dist_centers = sqrtf(powf(V1[0],2) + powf(V1[1],2));

        float c = (r_1-r_2)/dist_centers;
        V1[0] /= dist_centers;
        V1[1] /= dist_centers;

        float normal [2] = {(V1[0]*c)-(V1[1]*sqrtf(1-powf(c,2))),(V1[0]*sqrtf(1-powf(c,2)))+(V1[1]*c)};

        if (isnan(normal[0])){
            return;
        }

        float tangent_1 [2] = {p_c1[0] + (r_1* normal[0]),p_c1[1] + (r_1* normal[1])};
        float tangent_2 [2] = {p_c2[0] + (r_2* normal[0]),p_c2[1] + (r_2* normal[1])};

        float V2 [2] = {tangent_2[0]-tangent_1[0],tangent_2[1]-tangent_1[1]};

        float p2_h [2] = {point1[0], point1[1]};
        float v1 [2] = {p2_h[0]-p_c1[0], p2_h[1]-p_c1[1]};
        float v2 [2] = {tangent_1[0]-p_c1[0], tangent_1[1]-p_c1[1]};

        float theta_1 = atan2f(v2[1],v2[0]) - atan2f(v1[1],v1[0]);

        if (theta_1>0){
            theta_1-=(PI*2);
        }

        float angle = point1[2] + (PI/2);

        float x_vals [150] = { };
        float y_vals [150] = { };
        float d_theta = theta_1/50;

        for (int i=0;i<50;i++){
            x_vals[i] = (abs(r_1) * cosf(angle+(i*d_theta))) + p_c1[0];
            y_vals[i] = (abs(r_1) * sinf(angle+(i*d_theta))) + p_c1[1];
        }


        float p3_h [2] = {point2[0], point2[1]};
        v1[0] = tangent_2[0]-p_c2[0];
        v1[1] = tangent_2[1]-p_c2[1];

        v2[0] = p3_h[0] - p_c2[0];
        v2[1] = p3_h[1] - p_c2[1];

        float theta_2 = atan2f(v2[1],v2[0]) - atan2f(v1[1],v1[0]);

        if (theta_2<0){
            theta_2+=(PI*2);
        }

        angle = atan2f((tangent_2[1]-p_c2[1]),(tangent_2[0]-p_c2[0]));

        d_theta = theta_2/50;

        for (int i=0;i<50;i++){
            x_vals[i+100] = (abs(r_2) * cosf(angle+(i*d_theta))) + p_c2[0];
            y_vals[i+100] = (abs(r_2) * sinf(angle+(i*d_theta))) + p_c2[1];
        }

        float d_x = (x_vals[99] - x_vals[49])/50;
        float d_y = (y_vals[99] - y_vals[49])/50;

        for (int i=0;i<50;i++){
            x_vals[i+50] = x_vals[49] + (i*d_x);
            y_vals[i+50] = y_vals[49] + (i*d_y);
        }

        bool collision = check_col(y_vals,x_vals,obstacles,num_obs);

        if (collision){
            return;
        }

        float cost = abs((r_1*theta_1)) + abs((r_2*theta_2)) + sqrtf(powf(V2[0],2) + powf(V2[1],2));

        if (cost> *curCost){
            return;
        }

        *curCost = cost;
        return;
    }

    __device__ bool computeDubinsCost(float &cost, float *point1, float *point2, float *obstacles, int num_obs){

        float curCost = cost;

        int r_min = 2;

        RSRcost(&curCost, point1, point2, r_min, obstacles, num_obs);
        LSLcost(&curCost, point1, point2, r_min, obstacles, num_obs);
        LSRcost(&curCost, point1, point2, r_min, obstacles, num_obs);
        RSLcost(&curCost, point1, point2, r_min, obstacles, num_obs);

        bool connected = curCost < cost;
        cost = connected ? curCost : cost;
        return connected;
    }


    __global__ void wavefront(int *G, int *open, float *cost, float *threshold, const int n){
        const int index = threadIdx.x;
        if(index > n){
            return;
        }
        
        G[index] = open[index] && cost[index] <= threshold[0] ? 1 : 0;
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

    __device__ bool dubinCost(float &cost, float *x, float *y){
        float new_cost = powf(x[0]-y[0],2) + powf(x[1]-y[1],2) + powf(x[2]-y[2],2);
        bool connected = new_cost < cost;
        cost = connected ? new_cost : cost;
        return connected;

    }

    __global__ void dubinConnection(float *cost, int *parent, int *x, int *y, float *states, int *open, int *unexplored, const int xSize, const int *ySize, float *obstacles, int *num_obs){
        const int index = threadIdx.x;
        if(index > xSize){
            return;
        }

        for(int i=0; i < ySize[0]; i++){
            bool connected = computeDubinsCost(cost[x[index]], &states[x[index]*3], &states[y[i]*3], obstacles, num_obs[0]);
            //bool connected = dubinCost(cost[x[index]], &states[x[index]*3], &states[y[i]*3]);
            parent[x[index]] = connected ? y[i]: parent[x[index]];
            cost[x[index]] = connected ? cost[y[i]] + cost[x[index]] : cost[x[index]];
            open[x[index]] = connected ? 1 : open[x[index]];
            open[y[i]] = connected ? 0 : open[y[i]];
            unexplored[x[index]] = connected ? 0 : unexplored[x[index]];
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
    states = np.array([[1,2,135], [0,3,30], [0,3,45], [1,2,-90], [1,2,-60], [0,3,-45]]).astype(np.float32)
    waypoints = np.array([0,1,2,3,4,5]).astype(np.int32)

    obstacles = np.array([[10,20,15,15], [-6,-3,-3,-6], [7,10,10,7]]).astype(np.float32)
    num_obs = np.array([3]).astype(np.int32)

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

    dev_obstacles = cuda.to_gpu(obstacles) 
    dev_num_obs = cuda.to_gpu(num_obs)

    dev_threshold = cuda.to_gpu(threshold)
    dev_n = cuda.to_gpu(np.array([n]))

    dev_neighbors = cuda.to_gpu(neighbors)
    dev_num_neighbors = cuda.to_gpu(num_neighbors)
    neighbors_index = cuda.to_gpu(num_neighbors)
    exclusiveScan(neighbors_index)
    print('neighbors index: ', neighbors_index)

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
        dev_ySize = dev_yscan[-1] + dev_open[-1]
        ySize = int(dev_ySize.get())

        dev_y = cuda.zeros(ySize, dtype=np.int32)
        compact(dev_y, dev_yscan, dev_open, dev_waypoints, dev_ySize, block=(n,1,1), grid=(1,1))
        
        dev_Gscan = cuda.to_gpu(dev_Gindicator)
        exclusiveScan(dev_Gscan)
        dev_gSize = dev_Gscan[-1] + dev_Gindicator[-1]
        gSize = int(dev_gSize.get())

        print('goal reached: ', goal_reached)
        print('y size: ', ySize, 'y: ' , dev_y)
        print('G size: ', gSize, 'G: ', dev_Gindicator)

        if ySize == 0 or i == 100:
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
        dev_xSize = dev_xscan[-1] + dev_xindicator[-1]
        xSize = int(dev_xSize.get())

        if xSize == 0:
            print('### x skip')
            continue

        dev_x = cuda.zeros(xSize, dtype=np.int32)
        compact(dev_x, dev_xscan, dev_xindicator, dev_waypoints, dev_xSize, block=(n,1,1), grid=(1,1))
        print('x size: ', xSize, 'x: ', dev_x)

        ######### connect neighbors ####################
        # # launch planning
        dubinConnection(dev_cost, dev_parent, dev_x, dev_y, dev_states, dev_open, dev_unexplored, dev_xSize, dev_ySize, dev_obstacles, dev_num_obs, block=(xSize,1,1), grid=(1,1))