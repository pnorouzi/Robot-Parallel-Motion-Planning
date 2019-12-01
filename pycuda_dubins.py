import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import numba

from pycuda.compiler import SourceModule
mod = SourceModule("""
#include <stdio.h>

__device__ bool check_col(float *y_vals,float *x_vals,float *obstacles, int num_obs){

  //printf ("obs_: %4.2f\\n",obstacles[0*4 +2]);
  //printf ("y_vals 89 %4.8f\\n",y_vals[89]);
  //printf ("x_vals 19 %4.8f\\n",x_vals[19]);
  //printf("num_obssss %d\\n",num_obs);

  for (int obs=0;obs<num_obs;obs++){
    for (int i=0;i<150;i++){

      if (obstacles[obs*4 +3]<=y_vals[i] && obstacles[obs*4 +1]>=y_vals[i]) {
        if (obstacles[obs*4]<=x_vals[i] && obstacles[obs*4 + 2]>=x_vals[i]){
          return true;
          }
      }
    }
  }
  return false;
}

__device__ void RSRcost(float *curCost, float *point1,float *point2, int r_min, float *obstacles, int num_obs)
{
  float PI = 3.141592653589793;

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
  float d_theta = theta_1/49;

  for (int i=0;i<50;i++)
  {
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

  //printf ("theta_2 %4.8f\\n",theta_2);
  d_theta = theta_2/49;

  for (int i=0;i<50;i++)
  {
    x_vals[i+100] = (abs(r_2) * cosf(angle+(i*d_theta))) + p_c2[0];
    y_vals[i+100] = (abs(r_2) * sinf(angle+(i*d_theta))) + p_c2[1];
  }

  float d_x = (x_vals[100] - x_vals[49])/49;
  float d_y = (y_vals[100] - y_vals[49])/49;

  for (int i=0;i<50;i++)
  {
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

__device__ void LSLcost(float *curCost, float *point1,float *point2, int r_min, float *obstacles, int num_obs)
{
  float PI = 3.141592653589793;

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
  float d_theta = theta_1/49;

  for (int i=0;i<50;i++)
  {
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

  d_theta = theta_2/49;
  //printf ("d_theta %4.8f\\n",d_theta);

  for (int i=0;i<50;i++)
  {
    x_vals[i+100] = (abs(r_2) * cosf(angle+(i*d_theta))) + p_c2[0];
    y_vals[i+100] = (abs(r_2) * sinf(angle+(i*d_theta))) + p_c2[1];
  }

  float d_x = (x_vals[100] - x_vals[49])/49;
  float d_y = (y_vals[100] - y_vals[49])/49;

  for (int i=0;i<50;i++)
  {
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

__device__ void LSRcost(float *curCost, float *point1,float *point2, int r_min, float *obstacles, int num_obs)
{
  float PI = 3.141592653589793;

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
  float d_theta = theta_1/49;

  for (int i=0;i<50;i++)
  {
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

  d_theta = theta_2/49;

  for (int i=0;i<50;i++)
  {
    x_vals[i+100] = (abs(r_2) * cosf(angle+(i*d_theta))) + p_c2[0];
    y_vals[i+100] = (abs(r_2) * sinf(angle+(i*d_theta))) + p_c2[1];
  }

  float d_x = (x_vals[100] - x_vals[49])/49;
  float d_y = (y_vals[100] - y_vals[49])/49;

  for (int i=0;i<50;i++)
  {
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

__device__ void RSLcost(float *curCost, float *point1,float *point2, int r_min, float *obstacles, int num_obs)
{
  float PI = 3.141592653589793;

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
  float d_theta = theta_1/49;

  for (int i=0;i<50;i++)
  {
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

  d_theta = theta_2/49;

  for (int i=0;i<50;i++)
  {
    x_vals[i+100] = (abs(r_2) * cosf(angle+(i*d_theta))) + p_c2[0];
    y_vals[i+100] = (abs(r_2) * sinf(angle+(i*d_theta))) + p_c2[1];
  }

  float d_x = (x_vals[100] - x_vals[49])/49;
  float d_y = (y_vals[100] - y_vals[49])/49;

  for (int i=0;i<50;i++)
  {
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

__device__ float computeDubinsCost(float *cost, float *point1, float *point2,float *obstacles,int *num_obs){

  float curCost = 999999999.0;

  

  int r_min = 2;


  RSRcost(&curCost,point1,point2,r_min,obstacles,num_obs[0]);
  LSLcost(&curCost,point1,point2,r_min,obstacles,num_obs[0]);
  LSRcost(&curCost,point1,point2,r_min,obstacles,num_obs[0]);
  RSLcost(&curCost,point1,point2,r_min,obstacles,num_obs[0]);

  return curCost;
    }

__global__ void main_test(float *cost, float *point1, float *point2,float *obstacles, int *num_obs)
{
  const int i = threadIdx.x;
  float outCost;
  outCost = computeDubinsCost(cost, point1, point2,obstacles, num_obs);

  *cost = outCost;
}
""")

main_test= mod.get_function("main_test")
cuda.Context.synchronize()

test_start =np.array([5,5,np.pi/2])
test_end = np.array([5,8,np.pi/3])
test_start = test_start.astype(np.float32)
test_end = test_end.astype(np.float32)

obstacles = np.array([[5,4,7,3]]).astype(np.float32)


#a = np.random.randn(400).astype(np.float32)
#b = np.random.randn(400).astype(np.float32)
num_obs = np.int32(obstacles.shape[0])

print(num_obs)
#print(length_n)
dest = np.zeros((1,)).astype(np.float32)
main_test(
        cuda.Out(dest), cuda.In(test_start), cuda.In(test_end),cuda.In(obstacles),cuda.In(num_obs),
        block=(400,1,1), grid=(1,1))
print(dest)
#cuda.Context.synchronize()

