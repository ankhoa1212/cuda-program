#pragma once

// CUDA launch params
#include "device_launch_parameters.h"

// utility and system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>

// CUDA libs
#include <cuda_runtime.h>
#include <curand.h>

// function definitions
int addVectors();
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);
float* generateRandomNumbers(int n, unsigned int seed);

__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void reduce(float* input_data, float* output_data) {
    extern __shared__ float shared_data[];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    shared_data[tid] = input_data[i];
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) output_data[blockIdx.x] = shared_data[0];
}

// default number of random numbers to generate
const int DEFAULT_RANDOM_NUMBERS = 4;

// default seed for random number generator
const unsigned int DEFAULT_SEED = 123;

// main function
int main()
{
    float* arr = new float[DEFAULT_RANDOM_NUMBERS];
    arr = generateRandomNumbers(DEFAULT_RANDOM_NUMBERS, DEFAULT_SEED);

    for (int i = 0; i < DEFAULT_RANDOM_NUMBERS; i++) {
        std::cout << "arr[" << i << "] = " << arr[i] << std::endl;
    }

    int blocks = 1;
    int threads = 256;

    printf("\nLaunching CUDA kernel with %i blocks and %i threads...\n", blocks, threads);
    reduce <<<blocks, threads>>> (arr, arr);
    
    cudaDeviceSynchronize();  // Wait for GPU to finish

    // TODO generate visualization of number generation as image
    // maybe add perlin noise generation

    for (int i = 0; i < DEFAULT_RANDOM_NUMBERS; i++) {
        std::cout << "arr[" << i << "] = " << arr[i] << std::endl;
    }

    printf("\nExiting main...");
    return 0;
}

// random number generator function
float* generateRandomNumbers(int n, unsigned int seed) {
    // create async stream for computation
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    // allocate space on GPU
    float* d_Rand;
    cudaMalloc((void**)&d_Rand, n * sizeof(float));

    printf("Seeding rng with %i ...\n", seed);
    curandGenerator_t prngGPU;
    curandCreateGenerator(&prngGPU, CURAND_RNG_PSEUDO_MTGP32);
    curandSetStream(prngGPU, stream);
    curandSetPseudoRandomGeneratorSeed(prngGPU, seed);
    
    // allocate space for results
    float* h_RandGPU;
    cudaMallocHost(&h_RandGPU, n * sizeof(float));

    printf("Generating %i random numbers on GPU...\n", DEFAULT_RANDOM_NUMBERS);
    curandGenerateUniform(prngGPU, (float*)d_Rand, n);
    
    printf("Reading back the results...\n\n");
    cudaMemcpyAsync(h_RandGPU, d_Rand, n * sizeof(float),
        cudaMemcpyDeviceToHost, stream);

    return h_RandGPU;
}

// default starter function
int addVectors()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    
    return 0;
}

// helper function for using CUDA to add vectors in parallel
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel <<<1, size >>> (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}
