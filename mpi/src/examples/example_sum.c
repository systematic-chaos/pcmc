#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include "ctimer.h"

#define N 16384000
#define MASTER 0
#define PAD 400

// Sum of the `n` components of an `A` double vector
int main(int argc, char **argv) {
    int myRank, p;
    MPI_Status status;

    int i;
    double *A, *b;
    double partialSum, totalSum;

    double t1, t2, tucpu, tscpu;

    A = malloc(N * sizeof(double));
    b = malloc(N * sizeof(double));

    srand(time(NULL));

    // MPI starts
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    MPI_Barrier(MPI_COMM_WORLD);
    if (myRank == MASTER) {
        printf("Sum application\n");
    }

    printf("Gonna start, I'm process %d\n", myRank);
    MPI_Barrier(MPI_COMM_WORLD);

    // Data generation
    // A is initialized in the master (0) processor
    if (myRank == MASTER) {
        for (i = 0; i < N; i++) {
            A[i] = (double)(rand() % PAD) + PAD;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Sequential program
    // Sequential sum in P0 (master)
    if (myRank == MASTER) {
        ctimer(&t1, &tucpu, &tscpu);
        totalSum = 0.;
        for (i = 0; i < N; i++) {
            totalSum += A[i];
        }
        ctimer(&t2, &tucpu, &tscpu);
        printf("Sequential sum =\t%f\n", totalSum);
        printf("Sequential computing time:\t%f\n", (float)(t2 - t1));
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Parallel program
    
    // Data distribution
    if (myRank == MASTER) {
        ctimer(&t1, &tucpu, &tscpu);
    }
    const int np = N / p;
    MPI_Scatter(A, np, MPI_DOUBLE, b, np, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    if (myRank == MASTER) {
        ctimer(&t2, &tucpu, &tscpu);
        printf("Distribution time:\t%f\n", (float)(t2 - t1));
    }

    
    // Distributed operations between processors
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (myRank == MASTER) {
        ctimer(&t1, &tucpu, &tscpu);
    }

    // Local sums
    partialSum = 0.;
    for (i = 0; i < np; i++) {
        partialSum += b[i];
    }
    printf("I'm process %d\n", myRank);
    printf("Partial sum =\t%f\n", partialSum);

    // Data gathering in P0 and total sum computation
    MPI_Reduce(&partialSum, &totalSum, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);

    if (myRank == MASTER) {
        ctimer(&t2, &tucpu, &tscpu);
    }
    MPI_Finalize();

    if (myRank == MASTER) {
        printf("Parallel sum =\t%f\n", totalSum);
        printf("Parallel computing time:\t%f\n", (float)(t2 - t1));
        printf("THAT'S ALL, FOLKS!\n");
    }

    return 0;
}
