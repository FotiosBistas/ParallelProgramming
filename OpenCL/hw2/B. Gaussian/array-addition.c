#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>

#define N 1024

// Kernel source code
const char *kernelSource =
    "__kernel void vectorAdd(__global const float* a, __global const float* b, __global float* c) {"
    "    int i = get_global_id(0);"
    "    c[i] = a[i] + b[i];"
    "}";

int main()
{
    // OpenCL setup
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue commandQueue;
    cl_program program;
    cl_kernel kernel;
    cl_mem bufferA, bufferB, bufferC;

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

    // Create the kernel
    kernel = clCreateKernel(program, "vectorAdd", &err);

    // Input data
    float a[N];
    float b[N];
    for (int i = 0; i < N; i++)
    {
        a[i] = (float)i;
        b[i] = (float)i;
    }

    // Create the buffers
    bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N, a, NULL);
    bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N, b, NULL);
    bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * N, NULL, NULL);

    // Set the kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&bufferC);

    // Enqueue the kernel
    size_t globalSize = N;
    cl_event event;
    clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, &event);

    // Wait for the kernel to finish
    clWaitForEvents(1, &event);

    // Calculate the execution time
    cl_ulong startTime, endTime;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, NULL);

    cl_ulong executionTime = endTime - startTime;
    printf("Execution Time: %lu nanoseconds\n", executionTime);

    // Read the result
    float c[N];
    clEnqueueReadBuffer(commandQueue, bufferC, CL_TRUE, 0, sizeof(float) * N, c, 0, NULL, NULL);

    // Clean up
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);

    return 0;
}
