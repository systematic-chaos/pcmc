#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include "ctimer.h"

#define N 16384000

/**
 * MPI program that computes the sum of the `n` elements of a double vector,
 * spanning a maximum of `p` processes, with and without using reduction instructions.
 * The root process must generate data, distribute them to other processes, gather partial
 * results and aggregate them into final results to be displayed.
 * Suggestion: Take `p` as power of 2 and `n` as multiple of `p`.
 */
int main(int argc, char **argv) {
    double *x, *xAux;
    double seqSum, parSum, parRedSum;
    int myRank, numProcessors, chunkSize;
    unsigned int i;
    double t1, t2, tucpu, tscpu;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcessors);
    chunkSize = N / numProcessors;

    // Data generation
    xAux = malloc(chunkSize * sizeof(double));
    if (!myRank) {
        x = malloc(N * sizeof(double));
        srand(time(NULL));
        for (i = 0; i < N; i++) {
            x[i] = rand() % (chunkSize / numProcessors);
        }
    }

    // Sequential problem solving by the root process
    if (!myRank) {
        seqSum = 0;
        ctimer(&t1, &tucpu, &tscpu);
        for (i = 0; i < N; i++) {
            seqSum += x[i];
        }
        ctimer(&t2, &tucpu, &tscpu);
        // Sequential results display
        printf("Sequential sum =\t%.0f\n", seqSum);
        printf("Time:\t%.9f seconds\n\n", (float)(t2 - t1));
    }

    // Data broadcast to the other processes
    ctimer(&t1, &tucpu, &tscpu);
    MPI_Scatter(x, chunkSize, MPI_DOUBLE, xAux, chunkSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Local computation in each processor
    parSum = 0;
    for (i = 0; i < chunkSize; i++) {
        parSum += xAux[i];
    }

    // Result gathering in the root process
    MPI_Reduce(&parSum, &parRedSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Gather(&parSum, 1, MPI_DOUBLE, xAux, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (!myRank) {
        parSum = 0;
        for (i = 0; i < numProcessors; i++) {
            parSum += xAux[i];
        }
    }
    ctimer(&t2, &tucpu, &tscpu);

    // Results display
    if (!myRank) {
        printf("Parallel sum (w reduction) =\t%.0f\n", parRedSum);
        printf("Parallel sum (w/ reduction) =\t%.0f\n", parSum);
        printf("Time:\t%.9f seconds\n", (float)(t2 - t1));
    }

    MPI_Finalize();
    return 0;
}
