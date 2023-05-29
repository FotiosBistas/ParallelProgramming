__kernel void gaussian_blur(__private int radius,
__constant unsigned char* input_red, 
__constant unsigned char* input_green, 
__constant unsigned char* input_blue,
__global unsigned char* output_red,
__global unsigned char* output_green,
__global unsigned char* output_blue)
{

    const int2 global_positions = {get_global_id(0),get_global_id(1)};
    const int2 global_sizes = {get_global_size(0),get_global_size(1)};

    float row,col; 
    float4 sums = {0.0, 0.0, 0.0, 0.0};

    float sigma = radius*radius;
    float expression = 1 / (3.14*2*sigma); 
    for (row=global_positions[0]-radius; row <=global_positions[0] + radius; row++)
    {
        for (col=global_positions[1]-radius; col <=global_positions[1]+ radius; col++) 
        {
            int x = clamp((int) col, 0, (int)global_sizes[0]-1);
            int y = clamp((int) row, 0, (int)global_sizes[1]-1);
            int tempPos = y *global_sizes[0]+ x;
            float square = (col-global_positions[1])*(col-global_positions[1])+(row-global_positions[0])*(row-global_positions[0]);
            float weight = exp(-square / (2*sigma)) * expression;

            sums[0] += input_red[tempPos] * weight;
            sums[1] += input_green[tempPos] * weight;
            sums[2] += input_blue[tempPos] * weight;
            sums[3] += weight;
        }    
    }

    output_red[global_positions[0]* global_sizes[0]+global_positions[1]] = round(sums[0]/sums[3]);
    output_green[global_positions[0]* global_sizes[0]+global_positions[1]] = round(sums[1]/sums[3]);
    output_blue[global_positions[0]* global_sizes[0]+global_positions[1]] = round(sums[2]/sums[3]);


}