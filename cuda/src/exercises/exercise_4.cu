/*
 * CUDA program that computes the product of a matrix of dimensions `m * n`
 * and a vector with `n` real elements expressed in simple precision.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define RAND_FLOAT() (rand() / (float)RAND_MAX)

#define M 64
#define N 256

// CUDA Kernel for the Device
__global__ void ex4MatVecMult(const float* A, const float* B, float* C) {
    const unsigned int matrixIndex = gridDim.x * threadIdx.x + blockIdx.x;
    const unsigned int vectorIndex = blockIdx.x;
    C[matrixIndex] = A[matrixIndex] * B[vectorIndex];
}

float checkMatVecMult(const float* A, const float* B, const float* C) {
    float accError = 0;
    for (int i = 0; i < M; i++) {
        for (unsigned int j = 0; j < N; j++) {
            accError += fabs(A[i * N + j] * B[j] - C[i * N + j]);
        }
    }
    return accError;
}

// Main program in the Host
int main(void) {

    // Allocate memory in the Host for the input matrices and vector, and the output matrix
    const size_t vectorMemorySize = N * sizeof(float);
    const size_t matrixMemorySize = M * vectorMemorySize;
    float *h_A = (float*)malloc(matrixMemorySize);
    float *h_B = (float*)malloc(vectorMemorySize);
    float *h_C = (float*)malloc(matrixMemorySize);

    // Initialize the input matrix and vector in the Host
    for (unsigned int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            h_A[i * M + j] = RAND_FLOAT();
        }
        h_B[i] = rand() / RAND_FLOAT();
    }

    // Allocate memory in the Device for the input matrices and vector, and the output matrix
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    cudaMalloc(&d_A, matrixMemorySize);
    cudaMalloc(&d_B, vectorMemorySize);
    cudaMalloc(&d_C, matrixMemorySize);

    // Copy the input matrix and vector A & B from the Host's memory to the Device's
    cudaMemcpy(d_A, h_A, matrixMemorySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, vectorMemorySize, cudaMemcpyHostToDevice);

    // Launch the CUDA Kernel
    const int threadsPerBlock = M;
    const int blocksPerGrid = N;
    printf("Multiplying a %dx%d matrix and a vector of %d elements.\n", M, N, N);
    printf("Launching the CUDA Kernel with %d blocks of %d threads each.\n",
        blocksPerGrid, threadsPerBlock);
    ex4MatVecMult<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);

    // Copy the result output matrix from the Device's memory to the Host's
    cudaMemcpy(h_C, d_C, matrixMemorySize, cudaMemcpyDeviceToHost);

    // Free memory in the Device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Display result data
    float error = checkMatVecMult(h_A, h_B, h_C);
    printf("Cumulative error: %.4f\n", error);

    // Free memory in the Host
    free(h_A);
    free(h_B);
    free(h_C);

    // Reset the Device and exit
    cudaDeviceReset();
    return 0;
}
