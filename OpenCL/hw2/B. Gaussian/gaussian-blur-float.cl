__kernel void gaussian_blur(__private int radius,
__constant unsigned char* input_red, 
__constant unsigned char* input_green, 
__constant unsigned char* input_blue,
__global unsigned char* output_red,
__global unsigned char* output_green,
__global unsigned char* output_blue)
{

   int i = get_global_id(0); 
   int j = get_global_id(1);


   int width = get_global_size(0);
   int height = get_global_size(1);

   float row,col; 
   float weightSum = 0.0, redSum = 0.0, greenSum = 0.0, blueSum = 0.0;

   for (row = i-radius; row <= i + radius; row++)
   {
      for (col = j-radius; col <= j + radius; col++) 
      {
         int x = clamp((int) col, 0, (int) width-1);
         int y = clamp((int) row, 0, (int) height-1);
         int tempPos = y * width + x;
         float square = (col-j)*(col-j)+(row-i)*(row-i);
         float sigma = radius*radius;
         float weight = exp(-square / (2*sigma)) / (3.14*2*sigma);

         redSum += input_red[tempPos] * weight;
         greenSum += input_green[tempPos] * weight;
         blueSum += input_blue[tempPos] * weight;
         weightSum += weight;
      }    
   }
   output_red[i*width+j] = round(redSum/weightSum);
   output_green[i*width+j] = round(greenSum/weightSum);
   output_blue[i*width+j] = round(blueSum/weightSum);


}