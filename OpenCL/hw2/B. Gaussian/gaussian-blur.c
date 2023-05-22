#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <CL/cl.h>

#pragma pack(push, 2)          
	typedef struct bmpheader_ 
	{
		char sign;
		int size;
		int notused;
		int data;
		int headwidth;
		int width;
		int height;
		short numofplanes;
		short bitpix;
		int method;
		int arraywidth;
		int horizresol;
		int vertresol;
		int colnum;
		int basecolnum;
	} bmpheader_t;
#pragma pack(pop)

#define global 1024
#define work_group 256 

/* This is the image structure, containing all the BMP information 
 * plus the RGB channels.
 */
typedef struct img_
{
	bmpheader_t header;
	int rgb_width;
	unsigned char *imgdata;
	unsigned char *red;
	unsigned char *green;
	unsigned char *blue;
} img_t;

void gaussian_blur_serial(int, img_t *, img_t *);
void gaussian_blur_omp(int, img_t *, img_t *);
void gaussian_blur_opencl(int, img_t *, img_t *);


const char* img_folder = "producedImages/";

/* START of BMP utility functions */
static
void bmp_read_img_from_file(char *inputfile, img_t *img) 
{
	FILE *file;
	bmpheader_t *header = &(img->header);

	file = fopen(inputfile, "rb");
	if (file == NULL)
	{
		fprintf(stderr, "File %s not found; exiting.", inputfile);
		exit(1);
	}
	
	fread(header, sizeof(bmpheader_t)+1, 1, file);
	if (header->bitpix != 24)
	{
		fprintf(stderr, "File %s is not in 24-bit format; exiting.", inputfile);
		exit(1);
	}

	img->imgdata = (unsigned char*) calloc(header->arraywidth, sizeof(unsigned char));
	if (img->imgdata == NULL)
	{
		fprintf(stderr, "Cannot allocate memory for image data; exiting.");
		exit(1);
	}
	
	fseek(file, header->data, SEEK_SET);
	fread(img->imgdata, header->arraywidth, 1, file);
	fclose(file);
}

static
void bmp_clone_empty_img(img_t *imgin, img_t *imgout)
{
	imgout->header = imgin->header;
	imgout->imgdata = 
		(unsigned char*) calloc(imgout->header.arraywidth, sizeof(unsigned char));
	if (imgout->imgdata == NULL)
	{
		fprintf(stderr, "Cannot allocate memory for clone image data; exiting.");
		exit(1);
	}
}

static
void bmp_write_data_to_file(char *fname, img_t *img) 
{
	FILE *file;
	bmpheader_t *bmph = &(img->header);
	//custom modification for image folder 
	//all images created from the algorithm are written to the folder
	char* name_with_folder = malloc(strlen(fname) + strlen(img_folder) + 1);
	strcpy(name_with_folder, img_folder);
	strcat(name_with_folder, fname);
	//changed fname to name with folder 
	file = fopen(name_with_folder, "wb");
	fwrite(bmph, sizeof(bmpheader_t)+1, 1, file);
	fseek(file, bmph->data, SEEK_SET);
	fwrite(img->imgdata, bmph->arraywidth, 1, file);
	fclose(file);
	free(name_with_folder);
}

static
void bmp_rgb_from_data(img_t *img)
{
	bmpheader_t *bmph = &(img->header);

	int i, j, pos = 0;
	int width = bmph->width, height = bmph->height;
	int rgb_width = img->rgb_width;

	for (i = 0; i < height; i++) 
		for (j = 0; j < width * 3; j += 3, pos++)
		{
			img->red[pos]   = img->imgdata[i * rgb_width + j];
			img->green[pos] = img->imgdata[i * rgb_width + j + 1];
			img->blue[pos]  = img->imgdata[i * rgb_width + j + 2];  
		}
}

static
void bmp_data_from_rgb(img_t *img)
{
	bmpheader_t *bmph = &(img->header);
	int i, j, pos = 0;
	int width = bmph->width, height = bmph->height;
	int rgb_width = img->rgb_width;

	for (i = 0; i < height; i++ ) 
		for (j = 0; j < width* 3 ; j += 3 , pos++) 
		{
			img->imgdata[i * rgb_width  + j]     = img->red[pos];
			img->imgdata[i * rgb_width  + j + 1] = img->green[pos];
			img->imgdata[i * rgb_width  + j + 2] = img->blue[pos];
		}
}

static
void bmp_rgb_alloc(img_t *img)
{
	int width, height;

	width = img->header.width;
	height = img->header.height;

	img->red = (unsigned char*) calloc(width*height, sizeof(unsigned char));
	if (img->red == NULL)
	{
		fprintf(stderr, "Cannot allocate memory for the red channel; exiting.");
		exit(1);
	}

	img->green = (unsigned char*) calloc(width*height, sizeof(unsigned char));
	if (img->green == NULL)
	{
		fprintf(stderr, "Cannot allocate memory for the green channel; exiting.");
		exit(1);
	}

	img->blue = (unsigned char*) calloc(width*height, sizeof(unsigned char));
	if (img->blue == NULL)
	{
		fprintf(stderr, "Cannot allocate memory for the blue channel; exiting.");
		exit(1);
	}

	img->rgb_width = width * 3;
	if ((width * 3  % 4) != 0) {
	   img->rgb_width += (4 - (width * 3 % 4));  
	}
}

static
void bmp_img_free(img_t *img)
{
	free(img->red);
	free(img->green);
	free(img->blue);
	free(img->imgdata);
}

/* END of BMP utility functions */

/* check bounds */
int clamp(int i , int min , int max)
{
	if (i < min) return min;
	else if (i > max) return max;
	return i;  
}

/* Sequential Gaussian Blur */
void gaussian_blur_serial(int radius, img_t *imgin, img_t *imgout)
{
	int i, j;
	int width = imgin->header.width, height = imgin->header.height;
	double row, col;
	double weightSum = 0.0, redSum = 0.0, greenSum = 0.0, blueSum = 0.0;

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width ; j++) 
		{
			for (row = i-radius; row <= i + radius; row++)
			{
				for (col = j-radius; col <= j + radius; col++) 
				{
					int x = clamp(col, 0, width-1);
					int y = clamp(row, 0, height-1);
					int tempPos = y * width + x;
					double square = (col-j)*(col-j)+(row-i)*(row-i);
					double sigma = radius*radius;
					double weight = exp(-square / (2*sigma)) / (3.14*2*sigma);

					redSum += imgin->red[tempPos] * weight;
					greenSum += imgin->green[tempPos] * weight;
					blueSum += imgin->blue[tempPos] * weight;
					weightSum += weight;
				}    
			}
			imgout->red[i*width+j] = round(redSum/weightSum);
			imgout->green[i*width+j] = round(greenSum/weightSum);
			imgout->blue[i*width+j] = round(blueSum/weightSum);

			redSum = 0;
			greenSum = 0;
			blueSum = 0;
			weightSum = 0;
		}
	}
}


/* Parallel Gaussian Blur with OpenMP loop parallelization */
void gaussian_blur_omp_loops(int radius, img_t *imgin, img_t *imgout)
{
	/* TODO: Implement parallel Gaussian Blur using OpenMP loop parallelization */
	int i, j;
	int width = imgin->header.width, height = imgin->header.height;
	double row, col;
	double weightSum = 0.0, redSum = 0.0, greenSum = 0.0, blueSum = 0.0;
	
	
	#pragma omp parallel for schedule(dynamic) firstprivate(redSum,greenSum,blueSum,weightSum,radius,imgout,width,height) private(i,j,row,col) default(none) shared(imgin) if(height >= width)
	for (i = 0; i < height; i++)
	{
		
		#pragma omp parallel for schedule(dynamic) firstprivate(redSum,greenSum,blueSum,weightSum,radius,imgout,width,height,i) private(j,row,col) default(none) shared(imgin) if(height <  width)
		for (j = 0; j < width ; j++) 
		{
			for (row = i-radius; row <= i + radius; row++)
			{
				for (col = j-radius; col <= j + radius; col++) 
				{
					int x = clamp(col, 0, width-1);
					int y = clamp(row, 0, height-1);
					int tempPos = y * width + x;
					double square = (col-j)*(col-j)+(row-i)*(row-i);
					double sigma = radius*radius;
					double weight = exp(-square / (2*sigma)) / (3.14*2*sigma);

					redSum += imgin->red[tempPos] * weight;
					greenSum += imgin->green[tempPos] * weight;
					blueSum += imgin->blue[tempPos] * weight;
					weightSum += weight;
				}    
			}
			imgout->red[i*width+j] = round(redSum/weightSum);
			imgout->green[i*width+j] = round(greenSum/weightSum);
			imgout->blue[i*width+j] = round(blueSum/weightSum);

			redSum = 0;
			greenSum = 0;
			blueSum = 0;
			weightSum = 0;
		}
	}
}


/* Parallel Gaussian Blur with OpenMP tasks */
void gaussian_blur_omp_tasks(int radius, img_t *imgin, img_t *imgout)
{
	/* TODO: Implement parallel Gaussian Blur using OpenMP tasks */
	int i, j;
	int width = imgin->header.width, height = imgin->header.height;
	double row, col;
	double weightSum = 0.0, redSum = 0.0, greenSum = 0.0, blueSum = 0.0;
	
	#pragma omp parallel private(i,j,row,col) firstprivate(weightSum,redSum,greenSum,blueSum,radius,imgout,width,height) default(none) shared(imgin)
	#pragma omp single
	for (i = 0; i < height; i++)
	{

		#pragma omp task firstprivate(i,weightSum,redSum,greenSum,blueSum,radius,imgout,width,height) private(j,row,col) default(none) shared(imgin)
		{
			for (j = 0; j < width ; j++) 
			{
				for (row = i-radius; row <= i + radius; row++)
				{
					for (col = j-radius; col <= j + radius; col++) 
					{
						int x = clamp(col, 0, width-1);
						int y = clamp(row, 0, height-1);
						int tempPos = y * width + x;
						double square = (col-j)*(col-j)+(row-i)*(row-i);
						double sigma = radius*radius;
						double weight = exp(-square / (2*sigma)) / (3.14*2*sigma);

						redSum += imgin->red[tempPos] * weight;
						greenSum += imgin->green[tempPos] * weight;
						blueSum += imgin->blue[tempPos] * weight;
						weightSum += weight;
					}    
				}
				imgout->red[i*width+j] = round(redSum/weightSum);
				imgout->green[i*width+j] = round(greenSum/weightSum);
				imgout->blue[i*width+j] = round(blueSum/weightSum);

				redSum = 0;
				greenSum = 0;
				blueSum = 0;
				weightSum = 0;
			}
		}
	}//implicit barrier tasks must finish 
}

//helper function to read kernel from file 
size_t read_kernel_from_file(char** source_str, char* filename){
	//read the kernel from the kernel file 
	FILE *fp = NULL; 
	size_t source_size, program_size;
	// try to open file 
	fp = fopen(filename,"rb");
	if(!fp){
		fprintf(stderr, "Failed to load kernel: %s\n",filename);
		return -1; 
	}
	//seek the end of the file to determine kernel size
	fseek(fp, 0, SEEK_END); 
	program_size = ftell(fp);
	rewind(fp);
	//allocate memory for kernel
	*source_str = (char*)malloc(program_size + 1); 
	(*source_str)[program_size] = '\0';
	//read the kernel from the file 
	fread(*source_str, sizeof(char), program_size, fp); 
	fclose(fp);
	return program_size;
}

/* Parallel Gaussian Blur with OpenCL */
cl_ulong gaussian_blur_opencl_gpu(int radius, img_t *imgin, img_t *imgout)
{
	/* TODO: Implement parallel Gaussian Blur using OpenCL */

	char* kernelSource;

	if(read_kernel_from_file(&kernelSource, "gaussian-blur.cl") < 0){
		fprintf(stderr, "Error while calling read kernel from file"); 
		return 0; 
	};

	printf("Read kernel for GPU acceleration: \n %s\n", kernelSource);

    // OpenCL setup
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue commandQueue;
    cl_program program;
    cl_kernel kernel;


    cl_int err;

    // Create the OpenCL context
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

    // Create the command queue
    commandQueue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);

    // Create the program from the kernel source code
    program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, &err);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	
	//create the kernel 
	program = clCreateKernel(program, "gaussian-blur-gpu", &err);

	//allocate memory on the device 

	//spawn threads 
    size_t global_size = global; 
	size_t local_size = work_group; 
    cl_event event;
    clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, &event);

    // Wait for the kernel to finish
    clWaitForEvents(1, &event);

    // Calculate the execution time
    cl_ulong start_time, end_time;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);

    cl_ulong execution_time = end_time - start_time;

	free(kernelSource);
	//clean up
	clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);

	return execution_time; 
}

/* Parallel Gaussian Blur with OpenCL */
cl_ulong gaussian_blur_opencl_cpu(int radius, img_t *imgin, img_t *imgout)
{
	/* TODO: Implement parallel Gaussian Blur using OpenCL */

	char* kernelSource;

	if(read_kernel_from_file(&kernelSource, "gaussian-blur.cl") < 0){
		fprintf(stderr, "Error while calling read kernel from file"); 
		return 0; 
	};

	printf("Read kernel for GPU acceleration: \n %s\n", kernelSource);

    // OpenCL setup
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue commandQueue;
    cl_program program;
    cl_kernel kernel;


    cl_int err;

    // Create the OpenCL context
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

    // Create the command queue
    commandQueue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);

    // Create the program from the kernel source code
    program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, &err);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	
	//create the kernel 
	program = clCreateKernel(program, "gaussian-blur-gpu", &err);

	//allocate memory on the device 

	//spawn threads 
    size_t global_size = global; 
	size_t local_size = work_group; 
    cl_event event;
    clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, &event);

    // Wait for the kernel to finish
    clWaitForEvents(1, &event);

    // Calculate the execution time
    cl_ulong start_time, end_time;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);

    cl_ulong execution_time = end_time - start_time;

	free(kernelSource);

	//clean up
	clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);
	return execution_time; 
}



double timeit(void (*func)(), int radius, 
    img_t *imgin, img_t *imgout)
{
	struct timeval start, end;
	gettimeofday(&start, NULL);
	func(radius, imgin, imgout);
	gettimeofday(&end, NULL);
	return (double) (end.tv_usec - start.tv_usec) / 1000000 
		+ (double) (end.tv_sec - start.tv_sec);
}


char *remove_ext(char *str, char extsep, char pathsep) 
{
	char *newstr, *ext, *lpath;

	if (str == NULL) return NULL;
	if ((newstr = malloc(strlen(str) + 1)) == NULL) return NULL;

	strcpy(newstr, str);
	ext = strrchr(newstr, extsep);
	lpath = (pathsep == 0) ? NULL : strrchr(newstr, pathsep);
	if (ext != NULL) 
	{
		if (lpath != NULL) 
		{
			if (lpath < ext) 
				*ext = '\0';
		} 
		else 
			*ext = '\0';
	}
	return newstr;
}


int main(int argc, char *argv[]) 
{
	int i, j, radius;
	double exectime_serial = 0.0, exectime_omp_loops = 0.0, exectime_omp_tasks = 0.0, exectime_opencl_gpu = 0.0, exectime_opencl_cpu = 0.0;
	struct timeval start, stop; 
	char *inputfile, *noextfname;   
	char seqoutfile[128], paroutfile_loops[128], paroutfile_tasks[128], paroutfile_opencl_gpu[128], paroutfile_opencl_cpu[128];
	img_t imgin, imgout, pimgout_loops, pimgout_tasks, pimgout_opencl_gpu, pimgout_opencl_cpu;
	

	//custom modification for easier tests 4 arguments  
	if (argc < 4)
	{
		fprintf(stderr, "Syntax error not enough arguments were provided: %s <blur-radius> <filename> <num_threads>, \n\te.g. %s 2 500.bmp 3\n", 
			argv[0], argv[0]);
		fprintf(stderr, "Available images: 500.bmp, 1000.bmp, 1500.bmp\n");
		exit(1);
	}

	inputfile = argv[2];

	radius = atoi(argv[1]);
	if (radius < 0)
	{
		fprintf(stderr, "Radius should be an integer >= 0; exiting.");
		exit(1);
	}

	//custom modification for easier tests 
	//pass threads through the command line 
	int num_threads = atoi(argv[3]);

	if(num_threads < 0){
		fprintf(stderr, "Number of threads given to open mp should be larger than 0\n");
		exit(1);
	}

	//setting dynamic threads 
	printf("Setting dynamic threads to: %d\n",num_threads);
	omp_set_dynamic(0);//don't leave the thread number up to the runtime 
	omp_set_num_threads(num_threads);

	noextfname = remove_ext(inputfile, '.', '/');
	sprintf(seqoutfile, "%s-r%d-serial.bmp", noextfname, radius);
	sprintf(paroutfile_loops, "%s-r%d-omp-loops.bmp", noextfname, radius);
	sprintf(paroutfile_tasks, "%s-r%d-omp-tasks.bmp", noextfname, radius);
	//custom modification to display file with opencl 
	sprintf(paroutfile_opencl_gpu,"%s-r%d-opencl-gpu.bmp", noextfname, radius);
	sprintf(paroutfile_opencl_cpu, "%s-r%d-opencl-cpu.bmp", noextfname, radius);

	bmp_read_img_from_file(inputfile, &imgin);
	bmp_clone_empty_img(&imgin, &imgout);
	bmp_clone_empty_img(&imgin, &pimgout_loops);
	bmp_clone_empty_img(&imgin, &pimgout_tasks);
	//custom modification 
	bmp_clone_empty_img(&imgin, &pimgout_opencl_gpu);
	bmp_clone_empty_img(&imgin, &pimgout_opencl_cpu);
	bmp_rgb_alloc(&imgin);
	bmp_rgb_alloc(&imgout);
	bmp_rgb_alloc(&pimgout_loops);
	bmp_rgb_alloc(&pimgout_tasks);
	bmp_rgb_alloc(&pimgout_opencl_gpu);
	bmp_rgb_alloc(&pimgout_opencl_cpu);

	printf("<<< Gaussian Blur (h=%d,w=%d,r=%d) >>>\n", imgin.header.height, 
	       imgin.header.width, radius);

	/* Image data to R,G,B */
	bmp_rgb_from_data(&imgin);

	/* Run & time serial Gaussian Blur */
	exectime_serial = timeit(gaussian_blur_serial, radius, &imgin, &imgout);

	/* Save the results (serial) */
	bmp_data_from_rgb(&imgout);
	bmp_write_data_to_file(seqoutfile, &imgout);

	/* Run & time OpenMP Gaussian Blur (w/ loops) */
	exectime_omp_loops = timeit(gaussian_blur_omp_loops, radius, &imgin, &pimgout_loops);

	/* Save the results (parallel w/ loops) */
	bmp_data_from_rgb(&pimgout_loops);
	bmp_write_data_to_file(paroutfile_loops, &pimgout_loops);

	/* Run & time OpenMP Gaussian Blur (w/ tasks) */
	exectime_omp_tasks = timeit(gaussian_blur_omp_tasks, radius, &imgin, &pimgout_tasks);

	/* Save the results (parallel w/ tasks) */
	bmp_data_from_rgb(&pimgout_tasks);
	bmp_write_data_to_file(paroutfile_tasks, &pimgout_tasks);
		

	//custom modication to check time for OpenCL gpu
	exectime_opencl_gpu = gaussian_blur_opencl_gpu(radius, &imgin, &pimgout_opencl_gpu);

	/* Save the results (parallel w/ OpenCL) */
	bmp_data_from_rgb(&pimgout_opencl_gpu);
	bmp_write_data_to_file(paroutfile_opencl_gpu, &pimgout_opencl_gpu);

	//custom modication to check time for OpenCL cpu
	exectime_opencl_cpu = gaussian_blur_opencl_cpu(radius, &imgin, &pimgout_opencl_cpu);

	/* Save the results (parallel w/ OpenCL) */
	bmp_data_from_rgb(&pimgout_opencl_cpu);
	bmp_write_data_to_file(paroutfile_opencl_cpu, &pimgout_opencl_cpu);

	printf("Total execution time (sequential): %lf\n", exectime_serial);
	printf("Total execution time (omp loops): %lf\n", exectime_omp_loops);
	printf("Total execution time (omp tasks): %lf\n", exectime_omp_tasks);
	//custom modification to check OpenCL execution time 
	printf("Total execution time (OpenCL gpu): %lf\n",(double)exectime_opencl_gpu/1e9);
	printf("Total execution time (OpenCL cpu): %lf\n",(double)exectime_opencl_cpu/1e9);

	bmp_img_free(&imgin);
	bmp_img_free(&imgout);
	bmp_img_free(&pimgout_loops);
	bmp_img_free(&pimgout_tasks);
	bmp_img_free(&pimgout_opencl_gpu);
	bmp_img_free(&pimgout_opencl_cpu);

	return 0;
}
