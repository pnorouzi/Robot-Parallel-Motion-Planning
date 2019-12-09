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

mod = SourceModule("""
    #include <stdio.h>
    #define ZERO 1e-6

    __device__ bool check_col(float *y_vals, float *x_vals, float *obstacles, int num_obs){
        if (num_obs==0){
            return false;
        }
        for (int obs=0;obs<num_obs;obs++){
            for (int i=0;i<150;i++){
                float min_y = fmin(obstacles[obs*4 +3],obstacles[obs*4 +1]);
                float max_y = fmax(obstacles[obs*4 +3],obstacles[obs*4 +1]);
                float min_x = fmin(obstacles[obs*4],obstacles[obs*4 +2]);
                float max_x = fmax(obstacles[obs*4],obstacles[obs*4 +2]);
                if (max_y>=y_vals[i] && min_y<=y_vals[i]) {
                    if (max_x>=x_vals[i] && min_x<=x_vals[i]){
                        return true;
                    }
                }
            }
        }
        return false;
    }


    __device__ void RSRcost(float *curCost, float *start_point, float *end_point, int r_min, float *obstacles, int num_obs){
        float PI = 3.141592653589793;

        float p_c1 [2] = { start_point[0] + (r_min * cosf(start_point[2] - PI/2)), start_point[1] + (r_min * sinf(start_point[2] - PI/2))}; 
        float p_c2 [2] = { end_point[0] + (r_min * cosf(end_point[2] - PI/2)), end_point[1] + (r_min * sinf(end_point[2] - PI/2))};

        float r_1 = sqrtf(powf(p_c1[0]-start_point[0],2.0) + powf(p_c1[1]-start_point[1],2.0));
        float r_2 = sqrtf(powf(p_c2[0]-end_point[0],2.0) + powf(p_c2[1]-end_point[1],2.0));

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


        float p2_h [2] = {start_point[0], start_point[1]};
        float v1 [2] = {p2_h[0]-p_c1[0], p2_h[1]-p_c1[1]};
        float v2 [2] = {tangent_1[0]-p_c1[0], tangent_1[1]-p_c1[1]};

        float theta_1 = atan2f(v2[1],v2[0]) - atan2f(v1[1],v1[0]);

        if (theta_1>ZERO){
            theta_1-=(PI*2);
        }

        float angle = start_point[2] + (PI/2);

        float x_vals [150] = { };
        float y_vals [150] = { };
        float d_theta = theta_1/49;

        for (int i=0;i<50;i++){
            x_vals[i] = (abs(r_1) * cosf(angle+(i*d_theta))) + p_c1[0];
            y_vals[i] = (abs(r_1) * sinf(angle+(i*d_theta))) + p_c1[1];
        }


        float p3_h [2] = {end_point[0], end_point[1]};
        v1[0] = tangent_2[0]-p_c2[0];
        v1[1] = tangent_2[1]-p_c2[1];

        v2[0] = p3_h[0] - p_c2[0];
        v2[1] = p3_h[1] - p_c2[1];

        float theta_2 = atan2f(v2[1],v2[0]) - atan2f(v1[1],v1[0]);


        if (theta_2>ZERO){
        theta_2-=(PI*2);
        }

        angle = atan2f((tangent_2[1]-p_c2[1]),(tangent_2[0]-p_c2[0]));

        d_theta = theta_2/49;

        for (int i=0;i<50;i++){
            x_vals[i+100] = (abs(r_2) * cosf(angle+(i*d_theta))) + p_c2[0];
            y_vals[i+100] = (abs(r_2) * sinf(angle+(i*d_theta))) + p_c2[1];
        }

        float d_x = (x_vals[100] - x_vals[49])/49;
        float d_y = (y_vals[100] - y_vals[49])/49;

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

        *curCost = fminf(cost, *curCost);
        return;
    }


    __device__ void LSLcost(float *curCost, float *start_point, float *end_point, int r_min, float *obstacles, int num_obs){
        float PI = 3.141592653589793;

        float p_c1 [2] = { start_point[0] + (r_min * cosf(start_point[2] + PI/2)), start_point[1] + (r_min * sinf(start_point[2] + PI/2))}; 
        float p_c2 [2] = { end_point[0] + (r_min * cosf(end_point[2] + PI/2)), end_point[1] + (r_min * sinf(end_point[2] + PI/2))};

        float r_1 = -1.0 * sqrtf(powf(p_c1[0]-start_point[0],2.0) + powf(p_c1[1]-start_point[1],2.0));
        float r_2 = -1.0 * sqrtf(powf(p_c2[0]-end_point[0],2.0) + powf(p_c2[1]-end_point[1],2.0));

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


        float p2_h [2] = {start_point[0], start_point[1]};
        float v1 [2] = {p2_h[0]-p_c1[0], p2_h[1]-p_c1[1]};
        float v2 [2] = {tangent_1[0]-p_c1[0], tangent_1[1]-p_c1[1]};

        float theta_1 = atan2f(v2[1],v2[0]) - atan2f(v1[1],v1[0]);

        if (theta_1<-ZERO){
            theta_1+=(PI*2);
        }

        float angle = start_point[2] - (PI/2);

        float x_vals [150] = { };
        float y_vals [150] = { };
        float d_theta = theta_1/49;

        for (int i=0;i<50;i++){
            x_vals[i] = (abs(r_1) * cosf(angle+(i*d_theta))) + p_c1[0];
            y_vals[i] = (abs(r_1) * sinf(angle+(i*d_theta))) + p_c1[1];
        }

        float p3_h [2] = {end_point[0], end_point[1]};
        v1[0] = tangent_2[0]-p_c2[0];
        v1[1] = tangent_2[1]-p_c2[1];

        v2[0] = p3_h[0] - p_c2[0];
        v2[1] = p3_h[1] - p_c2[1];

        float theta_2 = atan2f(v2[1],v2[0]) - atan2f(v1[1],v1[0]);


        if (theta_2<-ZERO){
            theta_2+=(PI*2);
        }
        angle = atan2f((tangent_2[1]-p_c2[1]),(tangent_2[0]-p_c2[0]));

        d_theta = theta_2/49;

        for (int i=0;i<50;i++){
            x_vals[i+100] = (abs(r_2) * cosf(angle+(i*d_theta))) + p_c2[0];
            y_vals[i+100] = (abs(r_2) * sinf(angle+(i*d_theta))) + p_c2[1];
        }

        float d_x = (x_vals[100] - x_vals[49])/49;
        float d_y = (y_vals[100] - y_vals[49])/49;

        for (int i=0;i<50;i++){
            x_vals[i+50] = x_vals[49] + (i*d_x);
            y_vals[i+50] = y_vals[49] + (i*d_y);
        }


        bool collision = check_col(y_vals,x_vals,obstacles,num_obs);

        if (collision){
            return;
        }


        float cost = abs((r_1*theta_1)) + abs((r_2*theta_2)) + sqrtf(powf(V2[0],2) + powf(V2[1],2));

        *curCost = fminf(cost, *curCost);
        return;
    }


    __device__ void LSRcost(float *curCost, float *start_point, float *end_point, int r_min, float *obstacles, int num_obs){
        float PI = 3.141592653589793;

        float p_c1 [2] = { start_point[0] + (r_min * cosf(start_point[2] + PI/2)), start_point[1] + (r_min * sinf(start_point[2] + PI/2))}; 
        float p_c2 [2] = { end_point[0] + (r_min * cosf(end_point[2] - PI/2)), end_point[1] + (r_min * sinf(end_point[2] - PI/2))};

        float r_1 = -1.0 * sqrtf(powf(p_c1[0]-start_point[0],2.0) + powf(p_c1[1]-start_point[1],2.0));
        float r_2 = sqrtf(powf(p_c2[0]-end_point[0],2.0) + powf(p_c2[1]-end_point[1],2.0));

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


        float p2_h [2] = {start_point[0], start_point[1]};
        float v1 [2] = {p2_h[0]-p_c1[0], p2_h[1]-p_c1[1]};
        float v2 [2] = {tangent_1[0]-p_c1[0], tangent_1[1]-p_c1[1]};

        float theta_1 = atan2f(v2[1],v2[0]) - atan2f(v1[1],v1[0]);

        if (theta_1<-ZERO){
            theta_1+=(PI*2);
        }

        float angle = start_point[2] - (PI/2);

        float x_vals [150] = { };
        float y_vals [150] = { };
        float d_theta = theta_1/49;

        for (int i=0;i<50;i++){
            x_vals[i] = (abs(r_1) * cosf(angle+(i*d_theta))) + p_c1[0];
            y_vals[i] = (abs(r_1) * sinf(angle+(i*d_theta))) + p_c1[1];
        }


        float p3_h [2] = {end_point[0], end_point[1]};
        v1[0] = tangent_2[0]-p_c2[0];
        v1[1] = tangent_2[1]-p_c2[1];

        v2[0] = p3_h[0] - p_c2[0];
        v2[1] = p3_h[1] - p_c2[1];

        float theta_2 = atan2f(v2[1],v2[0]) - atan2f(v1[1],v1[0]);

        if (theta_2>ZERO){
            theta_2-=(PI*2);
        }

        angle = atan2f((tangent_2[1]-p_c2[1]),(tangent_2[0]-p_c2[0]));

        d_theta = theta_2/49;

        for (int i=0;i<50;i++){
            x_vals[i+100] = (abs(r_2) * cosf(angle+(i*d_theta))) + p_c2[0];
            y_vals[i+100] = (abs(r_2) * sinf(angle+(i*d_theta))) + p_c2[1];
        }

        float d_x = (x_vals[100] - x_vals[49])/49;
        float d_y = (y_vals[100] - y_vals[49])/49;

        for (int i=0;i<50;i++){
            x_vals[i+50] = x_vals[49] + (i*d_x);
            y_vals[i+50] = y_vals[49] + (i*d_y);
        }

        bool collision = check_col(y_vals,x_vals,obstacles,num_obs);

        if (collision){
            return;
        }

        float cost = abs((r_1*theta_1)) + abs((r_2*theta_2)) + sqrtf(powf(V2[0],2) + powf(V2[1],2));

        *curCost = fminf(cost, *curCost);
        return;
    }


    __device__ void RSLcost(float *curCost, float *start_point, float *end_point, int r_min, float *obstacles, int num_obs){
        float PI = 3.141592653589793;

        float p_c1 [2] = { start_point[0] + (r_min * cosf(start_point[2] - PI/2)), start_point[1] + (r_min * sinf(start_point[2] - PI/2))}; 
        float p_c2 [2] = { end_point[0] + (r_min * cosf(end_point[2] + PI/2)), end_point[1] + (r_min * sinf(end_point[2] + PI/2))};

        float r_1 = sqrtf(powf(p_c1[0]-start_point[0],2.0) + powf(p_c1[1]-start_point[1],2.0));
        float r_2 = -1.0 * sqrtf(powf(p_c2[0]-end_point[0],2.0) + powf(p_c2[1]-end_point[1],2.0));

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

        float p2_h [2] = {start_point[0], start_point[1]};
        float v1 [2] = {p2_h[0]-p_c1[0], p2_h[1]-p_c1[1]};
        float v2 [2] = {tangent_1[0]-p_c1[0], tangent_1[1]-p_c1[1]};

        float theta_1 = atan2f(v2[1],v2[0]) - atan2f(v1[1],v1[0]);

        if (theta_1>ZERO){
            theta_1-=(PI*2);
        }

        float angle = start_point[2] + (PI/2);

        float x_vals [150] = { };
        float y_vals [150] = { };
        float d_theta = theta_1/49;

        for (int i=0;i<50;i++){
            x_vals[i] = (abs(r_1) * cosf(angle+(i*d_theta))) + p_c1[0];
            y_vals[i] = (abs(r_1) * sinf(angle+(i*d_theta))) + p_c1[1];
        }


        float p3_h [2] = {end_point[0], end_point[1]};
        v1[0] = tangent_2[0]-p_c2[0];
        v1[1] = tangent_2[1]-p_c2[1];

        v2[0] = p3_h[0] - p_c2[0];
        v2[1] = p3_h[1] - p_c2[1];

        float theta_2 = atan2f(v2[1],v2[0]) - atan2f(v1[1],v1[0]);

        if (theta_2<-ZERO){
            theta_2+=(PI*2);
        }

        angle = atan2f((tangent_2[1]-p_c2[1]),(tangent_2[0]-p_c2[0]));

        d_theta = theta_2/49;

        for (int i=0;i<50;i++){
            x_vals[i+100] = (abs(r_2) * cosf(angle+(i*d_theta))) + p_c2[0];
            y_vals[i+100] = (abs(r_2) * sinf(angle+(i*d_theta))) + p_c2[1];
        }

        float d_x = (x_vals[100] - x_vals[49])/49;
        float d_y = (y_vals[100] - y_vals[49])/49;

        for (int i=0;i<50;i++){
            x_vals[i+50] = x_vals[49] + (i*d_x);
            y_vals[i+50] = y_vals[49] + (i*d_y);
        }

        bool collision = check_col(y_vals,x_vals,obstacles,num_obs);

        if (collision){
            return;
        }

        float cost = abs((r_1*theta_1)) + abs((r_2*theta_2)) + sqrtf(powf(V2[0],2) + powf(V2[1],2));

        *curCost = fminf(cost, *curCost);
        return;
    }


    __device__ bool computeDubinsCost(float &cost, float &parentCost, float *end_point, float *start_point, float r_min, float *obstacles, int num_obs){
        float curCost = 9999999999.9;

        RSRcost(&curCost, start_point, end_point, r_min, obstacles, num_obs);
        LSLcost(&curCost, start_point, end_point, r_min, obstacles, num_obs);
        LSRcost(&curCost, start_point, end_point, r_min, obstacles, num_obs);
        RSLcost(&curCost, start_point, end_point, r_min, obstacles, num_obs);

        curCost += parentCost;
        bool connected = curCost < cost;

        cost = fminf(curCost, cost);
        return connected;
    }


    __global__ void dubinConnection(float *cost, int *parent, int *x, int *y, float *states, int *open, int *unexplored, const int *xSize, const int *ySize, float *obstacles, int *num_obs, float *radius){
        const int index = threadIdx.x + (blockIdx.x * blockDim.x);
        if(index >= xSize[0]){
            return;
        }

        for(int i=0; i < ySize[0]; i++){
            bool connected = computeDubinsCost(cost[x[index]], cost[y[i]], &states[x[index]*3], &states[y[i]*3], radius[0], obstacles, num_obs[0]);

            parent[x[index]] = connected ? y[i]: parent[x[index]];
            open[x[index]] = connected ? 1 : open[x[index]];
            open[y[i]] = 0;
            //open[y[i]] = connected ? 0 : open[y[i]];
            unexplored[x[index]] = connected ? 0 : unexplored[x[index]];
        }
    }


    __global__ void wavefront(int *G, int *open, float *cost, float *threshold, const int *n){
        const int index = threadIdx.x + (blockIdx.x * blockDim.x);
        if(index >= n[0]){
            return;
        }
        
        G[index] = open[index] && cost[index] <= threshold[0] ? 1 : 0;
    }


    __global__ void neighborIndicator(int *x_indicator, int *G, int *unexplored, int *neighbors, int *num_neighbors, int *neighbors_index, const int *n){
        const int index = threadIdx.x + (blockIdx.x * blockDim.x);
        if(index >= n[0]){
            return;
        }

        for(int i=0; i < num_neighbors[G[index]]; i++){
            int j = neighbors[neighbors_index[G[index]] + i];
            x_indicator[j] = unexplored[j] || x_indicator[j] > 0 ? 1 : 0;
        }      
    }


    __global__ void compact(int *x, int *scan, int *indicator, int *waypoints, const int *n){
        const int index = threadIdx.x + (blockIdx.x * blockDim.x);
        if(index >= n[0]){
            return;
        }

        if(indicator[index] == 1){
            x[scan[index]] = waypoints[index];
        }
    }

    __global__ void growThreshold(float *threshold, float *amount){
        const int index = threadIdx.x + (blockIdx.x * blockDim.x);
        if(index >= 1){
            return;
        }
        threshold[0] = threshold[0] + 2.0*amount[0];
    }
""")