__kernel void gaussian_blur(__global const float* a, __global const float* b, __global float* c)
{

   int i = get_global_id(0); 
   int j = get_global_id(1);

   c[i] = i + j;  
}