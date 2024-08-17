/*
 * CUDA program that computes the sum of two vectors with `n` real elements expressed
 * in simple precision, one of them multiplied by an `alpha` constant
 * (operation saxpy).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include "ctimer.h"

#define RAND_FLOAT() (rand() / (float)RAND_MAX)

#define ALPHA 4
#define N 1048576

// CUDA Kernel for the Device
__global__ void ex3VecSaxpy(const float* A, const float* B, float* C) {
    const unsigned int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N) {
        C[n] = B[n] + A[n] * ALPHA;
    }
}

float checkSaxpy(const float* A, const float* B, const float* C) {
    float accError = 0;
    for (unsigned int n = 0; n < N; n++) {
        accError += fabs(A[n] * ALPHA + B[n] - C[n]);
    }
    return accError;
}

// Main program in the Host
int main(void) {
    double t1, t2, tucpu, tscpu;

    // Allocate memory in the Host for the input and output vectors
    const size_t memorySize = N * sizeof(float);
    float *h_A = (float*)malloc(memorySize);
    float *h_B = (float*)malloc(memorySize);
    float *h_C = (float*)malloc(memorySize);
    
    // Initialize the input vectors in the Host
    srand(time(NULL));
    for (unsigned int i = 0; i < N; i++) {
        h_A[i] = RAND_FLOAT();
        h_B[i] = RAND_FLOAT();
    }

    // Allocate memory in the Device for the input and output vectors
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    cudaMalloc(&d_A, memorySize);
    cudaMalloc(&d_B, memorySize);
    cudaMalloc(&d_C, memorySize);

    // Copy the input vectors A & B from the Host's memory to the Device's
    cudaMemcpy(d_A, h_A, memorySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, memorySize, cudaMemcpyHostToDevice);

    // Launch the CUDA Kernel
    const int threadsPerBlock = 1024;
    const int blocksPerGrid = floorf((N + threadsPerBlock - 1) / threadsPerBlock);
    printf("Saxpy with two vectors of %d elements each.\n", N);
    printf("Launching the CUDA Kernel with %d blocks of %d threads each.\n",
        blocksPerGrid, threadsPerBlock);
    ctimer(&t1, &tucpu, &tscpu);
    ex3VecSaxpy<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);
    ctimer(&t2, &tucpu, &tscpu);
    printf("GPU time:\t%.9f seconds\n", (float)(t2 - t1));
    
    // Copy the result output vector from the Device's memory to the Host's
    cudaMemcpy(h_C, d_C, memorySize, cudaMemcpyDeviceToHost);

    // Free memory in the Device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Display result data
    ctimer(&t1, &tucpu, &tscpu);
    float error = checkSaxpy(h_A, h_B, h_C);
    ctimer(&t2, &tucpu, &tscpu);
    printf("Cumulative error: %.4f\n", error);
    printf("CPU time:\t%.9f seconds\n", (float)(t2 - t1));

    // Free memory in the Host
    free(h_A);
    free(h_B);
    free(h_C);

    // Reset the Device and exit
    cudaDeviceReset();
    return 0;
}
