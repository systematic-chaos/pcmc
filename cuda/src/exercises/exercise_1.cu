/*
 * CUDA program that computes the sum of two vectors with `n` elements each.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define RAND_FLOAT() (rand() / (float)RAND_MAX)

#define N 1048576

// CUDA Kernel code for the Device
__global__ void ex1VecSum(const float* A, const float* B, float* C) {
    const unsigned int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N) {
        C[n] = A[n] + B[n];
    }
}

float checkVecSum(const float* A, const float* B, const float* C) {
    float accError = 0;
    for (unsigned int n = 0; n < N; n++) {
        accError = fabs(A[n] + B[n] - C[n]);
    }
    return accError;
}

// Main program in the Host
int main(void) {

    // Allocate memory in the Host for the input and output vectors
    size_t memorySize = N * sizeof(float);
    float *h_A = (float*)malloc(memorySize);
    float *h_B = (float*)malloc(memorySize);
    float *h_C = (float*)malloc(memorySize);
    memset(h_C, 0, memorySize);

    // Initialize the input vectors in the Host
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

    // Launch the CUDA Kernel `ex1VecSum`
    const int threadsPerBlock = 256;
    const int blocksPerGrid = floorf((N + threadsPerBlock - 1) / threadsPerBlock);
    printf("Adding %d elements!\n", N);
    printf("Launching the CUDA Kernel with %d blocks of %d threads each.\n",
        blocksPerGrid, threadsPerBlock);
    ex1VecSum<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);

    // Copy the result output vector from the Device's memory to the Host's
    cudaMemcpy(h_C, d_C, memorySize, cudaMemcpyDeviceToHost);

    // Free memory in the Device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Display result data
    printf("A[0-2]\t%f %f %f\n", h_A[0], h_A[1], h_A[2]);
    printf("B[0-2]\t%f %f %f\n", h_B[0], h_B[1], h_B[2]);
    printf("C[0-2]\t%f %f %f\n", h_C[0], h_C[1], h_C[2]);
    float error = checkVecSum(h_A, h_B, h_C);
    printf("Cumulative error: %f\n", error);

    // Free memory in the Host
    free(h_A);
    free(h_B);
    free(h_C);

    // Reset the Device and exit
    cudaDeviceReset();
    return 0;
}
