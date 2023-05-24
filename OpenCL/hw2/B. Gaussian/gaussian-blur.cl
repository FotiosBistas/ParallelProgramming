__kernel void gaussian_blur(__private int radius,__constant unsigned char* red, __constant unsigned char* green, __constant unsigned char* blue)
{

   int i = get_global_id(0); 
   int j = get_global_id(1);

   int width = get_global_size(0);
   int height = get_global_size(1);

   double row,col; 
	double weightSum = 0.0, redSum = 0.0, greenSum = 0.0, blueSum = 0.0;

   for (row = i-radius; row <= i + radius; row++)
	{
      for (col = j-radius; col <= j + radius; col++) 
      {
         int x = clamp((int) col, 0, (int) width-1);
         int y = clamp((int) row, 0, (int) height-1);
         int tempPos = y * width + x;
         double square = (col-j)*(col-j)+(row-i)*(row-i);
         double sigma = radius*radius;
         double weight = exp(-square / (2*sigma)) / (3.14*2*sigma);

         redSum += red[tempPos] * weight;
         greenSum += green[tempPos] * weight;
         blueSum += blue[tempPos] * weight;
         weightSum += weight;
      }    
   }
   //imgout->red[i*width+j] = round(redSum/weightSum);
   //imgout->green[i*width+j] = round(greenSum/weightSum);
   //imgout->blue[i*width+j] = round(blueSum/weightSum);

}