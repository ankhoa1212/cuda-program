#pragma once

// for error checking and testing
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// utility and system includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <curand.h>

__global__ void addKernel(int* c, const int* a, const int* b);
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);
float* generateRandomNumbers(int n, unsigned int seed);