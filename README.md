# ParallelProgramming
This repository contains a collection of exercises from the Parallel Programming class taught by Andreas Basilakhs (AUEB 2023). The exercises focus on two widely used parallel programming frameworks: OpenMP and OpenCL. In this introduction, an overview of OpenMP, OpenCL is provided and detailed reports can be found in the respective directories.

# OpenMP

OpenMP, which stands for Open Multi-Processing, is an application programming interface (API) that supports shared memory multiprocessing programming in C, C++, and Fortran. It provides a simple and portable way to parallelize code across multiple cores or processors in a shared memory system. OpenMP allows developers to specify parallel regions, which are portions of code that can be executed concurrently by multiple threads.

By using directives, programmers can easily identify parallelizable sections of code and specify how the workload should be divided among threads. OpenMP also offers a range of synchronization and data-sharing mechanisms to control the behavior of parallel execution. It simplifies the process of writing parallel programs, enabling developers to achieve significant speedup on multi-core systems.

You can find a detailed report of the exercise set in the OpenMP directory. 

# OpenCL

OpenCL, short for Open Computing Language, is an open standard for heterogeneous computing that enables software developers to harness the computational power of different processing devices, such as CPUs, GPUs, and FPGAs. Unlike OpenMP, which focuses on shared memory systems, OpenCL is designed to work across a wide range of devices with different memory architectures.

With OpenCL, developers can write programs using a variant of the C programming language to execute tasks on multiple devices simultaneously. It provides a unified programming model that abstracts the underlying hardware, allowing efficient utilization of various computational resources. OpenCL's versatility makes it suitable for a variety of applications, including scientific simulations, image processing, and machine learning.

Within the realm of OpenCL, it is important to note that the exercises included in this repository were executed on a GPU, due to the CPU not being recognized by OpenCL (Graphics Processing Unit). GPUs excel at parallel computation due to their high number of cores and specialized architecture optimized for parallel workloads. By utilizing the capabilities of the GPU, you can explore the potential for significant speedup and improved performance in your parallel programming endeavors. 

You can find a detailed report in the OpenCL directory. 
