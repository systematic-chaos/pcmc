#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <math.h>
#include "ctimer.h"

#define N 65536000

/**
 * MPI program that computes the euclidean norm of a float vector
 * containing `n` elements, spanning a maximum of `p` processes.
 * Suggestion: It must not fail it the euclidean norm is smaller than the maximum
 * positive number that can be represented using [floating] simple precision.
 * The root process must generate data, distribute them to ther processes, and
 * retrieve, gather and display the final results.
 */
int main(int argc, char **argv) {
    float *x, *xAux;
    double norm2 = 0, mpiNorm2 = 0;
    int myRank, numProcessors, chunkSize;
    unsigned int i;
    double t1, t2, tucpu, tscpu;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcessors);
    chunkSize = N / numProcessors;

    // Data generation
    xAux = malloc(chunkSize * sizeof(float));
    if (!myRank) {
        x = malloc(N * sizeof(float));
        srand(time(NULL));
        for (i = 0; i < N; i++) {
            x[i] = rand() % (chunkSize / numProcessors);
        }
    }

    // Sequential problem solving by the root process
    if (!myRank) {
        ctimer(&t1, &tucpu, &tscpu);
        for (i = 0; i < N; i++) {
            norm2 += x[i] * x[i];
        }
        norm2 = sqrt(norm2);
        ctimer(&t2, &tucpu, &tscpu);
        // Sequential results display
        printf("Sequential euclidean norm =\t%.1f\n", norm2);
        printf("Time:\t%.9f seconds\n", (float)(t2 - t1));
    }

    // Data broadcast to other processes
    ctimer(&t1, &tucpu, &tscpu);
    MPI_Scatter(x, chunkSize, MPI_FLOAT, xAux, chunkSize, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Local computation in each processor
    for (i = 0; i < chunkSize; i++) {
        mpiNorm2 += xAux[i] * xAux[i];
    }

    // Result gathering in the root processor
    MPI_Reduce(&mpiNorm2, &norm2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    norm2 = sqrt(norm2);
    ctimer(&t2, &tucpu, &tscpu);

    // Results display
    if (!myRank) {
        printf("Parallel euclidean norm =\t%.1f\n", norm2);
        printf("Time:\t%.9f seconds\n", (float)(t2 - t1));
    }

    MPI_Finalize();
    return 0;
}
