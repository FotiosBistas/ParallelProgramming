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

const char *getErrorString(cl_int error)
{
switch(error){
    // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
    }
}

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

int OMP_THREAD_NUMBER = 0; 
int OMP_SET_DYNAMIC = 0;
const char* IMG_FOLDER = "producedImages/";

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
	char* name_with_folder = malloc(strlen(fname) + strlen(IMG_FOLDER) + 1);
	strcpy(name_with_folder, IMG_FOLDER);
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
	
	omp_set_dynamic(OMP_SET_DYNAMIC);
	omp_set_num_threads(OMP_THREAD_NUMBER);
	#pragma omp parallel for schedule(dynamic) firstprivate(redSum,greenSum,blueSum,weightSum,radius,imgout,width,height) private(i,j,row,col) default(none) shared(imgin) if(height >= width) 
	for (i = 0; i < height; i++)
	{
		
		#pragma omp parallel for schedule(dynamic) firstprivate(redSum,greenSum,blueSum,weightSum,radius,imgout,width,height,i) private(j,row,col) default(none) shared(imgin) if(height < width)
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
	
	omp_set_dynamic(OMP_SET_DYNAMIC);
	omp_set_num_threads(OMP_THREAD_NUMBER);
	omp_set_dynamic(0);
	#pragma omp parallel private(i,j,row,col) firstprivate(weightSum,redSum,greenSum,blueSum,radius,imgout,width,height) default(none) shared(imgin)
	#pragma omp single
	for (i = 0; i < height; i++)
	{

		#pragma omp task firstprivate(i,weightSum,redSum,greenSum,blueSum,radius,imgout,width,height) private(j,row,col) default(none) shared(imgin)
		{
			for (j = 0; j < width ; j++) 
			{
				#pragma omp task firstprivate(i,radius,imgout,width,height,j) private(row,col) default(none) shared(imgin,weightSum,redSum,greenSum,blueSum)
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
				}	
				#pragma omp taskwait //wait for the task below the j to finish as it is essentatial for completing the algorithm 
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

/* Helper function to read kernel from file*/
size_t read_kernel_from_file(char** source_str, char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Failed to open kernel file: %s\n", filename);
        return 0;
    }

    fseek(fp, 0, SEEK_END);
    size_t program_size = ftell(fp);
    rewind(fp);

    *source_str = (char*)malloc(program_size + 1);
    if (!(*source_str)) {
        fprintf(stderr, "Failed to allocate memory for kernel\n");
        fclose(fp);
        return 0;
    }

    size_t read_size = fread(*source_str, sizeof(char), program_size, fp);
    if (read_size != program_size) {
        fprintf(stderr, "Failed to read kernel from file: %s\n", filename);
        free(*source_str);
        fclose(fp);
        return 0;
    }

    (*source_str)[program_size] = '\0';

    fclose(fp);
    return program_size;
}

/*Release open cl memory after program execution or an error occurs*/
void release_cl_memory(cl_kernel* kernel, 
cl_program* program, 
cl_context* context,
cl_command_queue* command_queue)
{
	if(kernel){
		printf("Releasing kernel\n");
		clReleaseKernel(*kernel);
	}
	if(program){
		printf("Releasing program\n");
		clReleaseProgram(*program);
	}
	if(command_queue){
		printf("Releasing command queue\n");
  		clReleaseCommandQueue(*command_queue);
	}
	if(context){
		printf("Releasing context\n");
  		clReleaseContext(*context);
	}	
}


cl_ulong gaussian_blur_opencl_gpu(int radius, img_t *imgin, img_t *imgout)
{
    /* TODO: Implement parallel Gaussian Blur using OpenCL */

    char* kernelSource;

    if (read_kernel_from_file(&kernelSource, "gaussian-blur-vectors-math.cl") <= 0) {
        fprintf(stderr, "Error while calling read kernel from file\n");
        return 0;
    };

    printf("Read kernel for GPU acceleration: \n %s\n", kernelSource);

    cl_ulong startTime, endTime = 0;
    // OpenCL setup
    cl_platform_id platform;
    cl_device_id device;
    cl_context context = NULL;
    cl_command_queue commandQueue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;


    cl_int err;

    // Create the OpenCL context
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to get OpenCL platform IDs: %s\n", getErrorString(err));
		goto FAIL;
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to get OpenCL device IDs: %s\n", getErrorString(err));
		goto FAIL;
    }

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create OpenCL context: %s\n", getErrorString(err));
		goto FAIL;
    }

    // Create the command queue
    commandQueue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create OpenCL command queue: %s\n", getErrorString(err));
        
		goto FAIL;

    }

    // Create the program from the kernel source code
    program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create OpenCL program: %s\n", getErrorString(err));
        
		goto FAIL; 
    }

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to build OpenCL program: %s\n", getErrorString(err));

        // Print the build log
        char buildLog[4096];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
        fprintf(stderr, "Build log:\n%s\n", buildLog);

		goto FAIL;
    }

    // Create the kernel
    kernel = clCreateKernel(program, "gaussian_blur", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create OpenCL kernel: %s\n", getErrorString(err));
		goto FAIL;
    }

    // Create the buffers
    size_t red_size = sizeof(unsigned char) * (imgin->header.width * imgin->header.height);
    size_t green_size = sizeof(unsigned char) * (imgin->header.width * imgin->header.height);
    size_t blue_size = sizeof(unsigned char) * (imgin->header.width * imgin->header.height);


	printf("Input red size: %d, Input green size: %d, Input blue size: %d\n", red_size, green_size, blue_size);

    cl_mem bufferInputRed = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, red_size, imgin->red, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create buffer for input red: %s\n", getErrorString(err));
		goto FAIL;
    }
    cl_mem bufferInputGreen = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, green_size, imgin->green, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create buffer for input green: %s\n", getErrorString(err));
		goto FAIL;
    }
    cl_mem bufferInputBlue = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, blue_size, imgin->blue, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create buffer for input blue: %s\n", getErrorString(err));
		goto FAIL;
    }

	
	printf("Output red size: %d, output green size: %d, output blue size: %d\n", red_size, green_size, blue_size);

    cl_mem bufferOutputRed = clCreateBuffer(context, CL_MEM_WRITE_ONLY, red_size, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create buffer for output red: %s\n", getErrorString(err));
		goto FAIL;
    }
    cl_mem bufferOutputGreen = clCreateBuffer(context, CL_MEM_WRITE_ONLY, green_size, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create buffer for output green: %s\n", getErrorString(err));
		goto FAIL;
    }
    cl_mem bufferOutputBlue = clCreateBuffer(context, CL_MEM_WRITE_ONLY, blue_size, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create buffer for output blue: %s\n", getErrorString(err));
		goto FAIL;
    }

    // Set the kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_int), (void*)&radius);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to set kernel argument: %s\n", getErrorString(err));
		goto FAIL;
    }
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&bufferInputRed);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to set kernel argument: %s\n", getErrorString(err));
		goto FAIL;
    }
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&bufferInputGreen);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to set kernel argument: %s\n", getErrorString(err));
		goto FAIL;
    }
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&bufferInputBlue);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to set kernel argument: %s\n", getErrorString(err));
		goto FAIL;
    }
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&bufferOutputRed);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to set kernel argument: %s\n", getErrorString(err));
		goto FAIL;
    }
    err = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&bufferOutputGreen);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to set kernel argument: %s\n", getErrorString(err));
		goto FAIL;
    }
    err = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)&bufferOutputBlue);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to set kernel arguments: %s\n", getErrorString(err));
		goto FAIL;
    }



    // Spawn threads
    const size_t globalWorkSize[2] = {imgin->header.width, imgin->header.height};

	//modify local size
	//preffered multiple of local work group 64 
	//max work group size is 256
	const size_t localWorkSize[2] = {2,128}; 

	//create the local memory 
	//err = clSetKernelArg(kernel, 7, sizeof(unsigned char) * (localWorkSize[0] * localWorkSize[1]), NULL);
	//if (err != CL_SUCCESS) {
	//	fprintf(stderr, "Failed to set kernel argument: %s\n", getErrorString(err));
	//goto FAIL;
	//}
	//err = clSetKernelArg(kernel, 8, sizeof(unsigned char) * (localWorkSize[0] * localWorkSize[1]), NULL);
	//if (err != CL_SUCCESS) {
	//	fprintf(stderr, "Failed to set kernel argument: %s\n", getErrorString(err));
	//goto FAIL;
	//}
	//err = clSetKernelArg(kernel, 9, sizeof(unsigned char) * (localWorkSize[0] * localWorkSize[1]), NULL);
	//if (err != CL_SUCCESS) {
	//	fprintf(stderr, "Failed to set kernel arguments: %s\n", getErrorString(err));
	//goto FAIL;
	//}

    cl_event kernelEvent;
    err = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &kernelEvent);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to enqueue OpenCL kernel: %s\n", getErrorString(err));
		goto FAIL;
    }

    // Read the output buffers
    err = clEnqueueReadBuffer(commandQueue, bufferOutputRed, CL_TRUE, 0, red_size, imgout->red, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to read output red buffer: %s\n", getErrorString(err));
		goto FAIL;
    }
    err = clEnqueueReadBuffer(commandQueue, bufferOutputGreen, CL_TRUE, 0, green_size, imgout->green, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to read output green buffer: %s\n", getErrorString(err));
		goto FAIL;
    }
    err = clEnqueueReadBuffer(commandQueue, bufferOutputBlue, CL_TRUE, 0, blue_size, imgout->blue, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to read output blue buffer: %s\n", getErrorString(err));
		goto FAIL;
    }

    // Wait for kernel completion
    clWaitForEvents(1, &kernelEvent);

    // Get profiling information
    clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, NULL);
    clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, NULL);

    // Clean up
    FAIL:release_cl_memory(&kernel, &program, &context, &commandQueue);
	free(kernelSource);
	clReleaseMemObject(bufferInputRed);
	clReleaseMemObject(bufferInputGreen);
	clReleaseMemObject(bufferInputBlue);
	clReleaseMemObject(bufferOutputRed);
	clReleaseMemObject(bufferOutputGreen);
	clReleaseMemObject(bufferOutputBlue);
    return (endTime - startTime);
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
	OMP_THREAD_NUMBER = atoi(argv[3]);

	if(OMP_THREAD_NUMBER <= 0){
		fprintf(stderr, "Number of threads given to open mp should be larger than 0\n");
		exit(1);
	}

	//setting dynamic threads 
	printf("Setting omp threads to: %d\n", OMP_THREAD_NUMBER);

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


	///* Run & time serial Gaussian Blur */
	//exectime_serial = timeit(gaussian_blur_serial, radius, &imgin, &imgout);

	///* Save the results (serial) */
	//bmp_data_from_rgb(&imgout);
	//bmp_write_data_to_file(seqoutfile, &imgout);

	/* Run & time OpenMP Gaussian Blur (w/ loops) */
	//exectime_omp_loops = timeit(gaussian_blur_omp_loops, radius, &imgin, &pimgout_loops);
	///* Save the results (parallel w/ loops) */
	//bmp_data_from_rgb(&pimgout_loops);
	//bmp_write_data_to_file(paroutfile_loops, &pimgout_loops);
	///* Run & time OpenMP Gaussian Blur (w/ tasks) */
	//exectime_omp_tasks = timeit(gaussian_blur_omp_tasks, radius, &imgin, &pimgout_tasks);
	///* Save the results (parallel w/ tasks) */
	//bmp_data_from_rgb(&pimgout_tasks);
	//bmp_write_data_to_file(paroutfile_tasks, &pimgout_tasks);
		

	//custom modication to check time for OpenCL gpu
	exectime_opencl_gpu = gaussian_blur_opencl_gpu(radius, &imgin, &pimgout_opencl_gpu);

	/* Save the results (parallel w/ OpenCL) */
	bmp_data_from_rgb(&pimgout_opencl_gpu);
	bmp_write_data_to_file(paroutfile_opencl_gpu, &pimgout_opencl_gpu);

	////custom modication to check time for OpenCL cpu
	//exectime_opencl_cpu = gaussian_blur_opencl_cpu(radius, &imgin, &pimgout_opencl_cpu);

	///* Save the results (parallel w/ OpenCL) */
	//bmp_data_from_rgb(&pimgout_opencl_cpu);
	//bmp_write_data_to_file(paroutfile_opencl_cpu, &pimgout_opencl_cpu);

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
