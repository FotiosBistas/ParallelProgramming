__kernel void gaussian_blur(__private int radius,
__constant unsigned char* input_red, 
__constant unsigned char* input_green, 
__constant unsigned char* input_blue,
__global unsigned char* output_red,
__global unsigned char* output_green,
__global unsigned char* output_blue,
__local unsigned char* local_red,
__local unsigned char* local_green,
__local unsigned char* local_blue)
{

    int i = get_global_id(0); 
    int j = get_global_id(1);

    //get global and local dimensions 
    int width = get_global_size(0);
    int height = get_global_size(1);
    int localSizeX = get_local_size(0);
    int localSizeY = get_local_size(1);


    // Load data from global memory to local memory
    int localX = get_local_id(0);
    int localY = get_local_id(1);
    //this is the linear index that correspond to array local_color (because it is one dimensional)
    //e.g. work group size 16X16 if localY=15 (last one) and localX=0 localPos = 15 * 16 + 0 = 240
    int localPos = localY * localSizeX + localX;
    //same idea here but works for global. Inside the brackets is the linear global position
    local_red[localPos] = input_red[j * width + i];
    local_green[localPos] = input_green[j * width + i];
    local_blue[localPos] = input_blue[j * width + i];

    // Synchronize to ensure all data is loaded into local memory
    barrier(CLK_LOCAL_MEM_FENCE);

    float row,col; 
    float weightSum = 0.0, redSum = 0.0, greenSum = 0.0, blueSum = 0.0;

    for (int row = localX - radius; row <= localX + radius; row++)
    {
        for (int col = localY - radius; col <= localY + radius; col++) 
        {
            int x = clamp(col, 0, localSizeX - 1);
            int y = clamp(row, 0, localSizeY - 1);
            int tempPos = y * localSizeX + x;
            float square = (col - localY) * (col - localY) + (row - localX) * (row - localX);
            float sigma = radius * radius;
            float weight = exp(-square / (2 * sigma)) / (3.14 * 2 * sigma);

            redSum += local_red[tempPos] * weight;
            greenSum += local_green[tempPos] * weight;
            blueSum += local_blue[tempPos] * weight;
            weightSum += weight;
        }    
    }

    // Wait for all work-items to finish accessing local memory
    barrier(CLK_LOCAL_MEM_FENCE);

    // Store the result to the output global memory
    output_red[j * width + i] = round(redSum / weightSum);
    output_green[j * width + i] = round(greenSum / weightSum);
    output_blue[j * width + i] = round(blueSum / weightSum);

}