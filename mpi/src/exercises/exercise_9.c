#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include "ctimer.h"

#define N 8

/**
 * MPI program that multiplies two long integers stored as a couple of vectors,
 * spanning a maximum of `p` processors.
 */
int main(int argc, char **argv) {
    char *av, *bv, *avAux, *bvAux;
    unsigned long a, b, aAux, bAux, mult;
    int myRank, numProcessors, chunkSize;
    unsigned int i, pow10;
    double t1, t2, tucpu, tscpu;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcessors);
    chunkSize = N / numProcessors;

    // Data generation
    if (!myRank) {
        av = malloc(N * sizeof(char));
        bv = malloc(N * sizeof(char));
        srand(time(NULL));
        for (i = 0; i < N; i++) {
            av[i] = rand() % 10;
            bv[i] = rand() % 10;
        }
    }
    avAux = malloc(chunkSize * sizeof(char));
    bvAux = malloc(chunkSize * sizeof(char));

    // Sequential problem solving by the root process
    if (!myRank) {
        ctimer(&t1, &tucpu, &tscpu);
        a = b = 0;
        for (i = 0; i < N; i++) {
            pow10 = pow(10, N - (i + 1));
            a += av[i] * pow10;
            b += bv[i] * pow10;
        }
        mult = a * b;
        ctimer(&t2, &tucpu, &tscpu);
        // Sequential results display
        printf("Sequential product =\t%ld\n", mult);
        printf("Time:\t%.9f seconds\n", (float)(t2 - t1));
    }

    // Data broadcast to other processes
    ctimer(&t1, &tucpu, &tscpu);
    MPI_Scatter(av, chunkSize, MPI_CHAR, avAux, chunkSize, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(bv, chunkSize, MPI_CHAR, bvAux, chunkSize, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Result gathering in the root processor
    aAux = bAux = 0;
    for (i = 0; i < chunkSize; i++) {
        pow10 = pow(10, N - (myRank * chunkSize + i + 1));
        aAux += avAux[i] * pow10;
        bAux += bvAux[i] * pow10;
    }
    MPI_Reduce(&aAux, &a, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&bAux, &b, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    mult = a * b;
    ctimer(&t2, &tucpu, &tscpu);

    // Results display
    if (!myRank) {
        printf("Parallel product =\t%ld\n", mult);
        printf("Time:\t%.9f seconds\n", (float)(t2 - t1));
    }

    MPI_Finalize();
    return 0;
}
