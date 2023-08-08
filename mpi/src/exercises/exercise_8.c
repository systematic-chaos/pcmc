#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include "ctimer.h"

#define M 3
#define C 3200
#define N 64

double polynom(double *f, int grade, double x);

/**
 * MPI program that computes the integral, between `a` and `b`, of a
 * polynomical function of grade `m`, splitting the interval `[a, b]`
 * in `n` subintervals and spanning a maximum of `p` processors.
 */
int main(int argc, char **argv) {
    double f[M + 1];
    const double a = 0., b = 30.;
    double integral, mpiIntegral, x;
    int myRank, numProcessors, i;
    double t1, t2, tucpu, tscpu;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcessors);

    const double xInterval = fabs(b - a) / N;
    const double xStart = a < b ? a : b;
    const int numBlocks = N / numProcessors;

    // Data generation
    if (!myRank) {
        srand(time(NULL));
        for (i = 0; i <= M; i++) {
            f[i] = rand() % (M * N + C);
        }
    }

    // Sequential problem solving by the root process
    if (!myRank) {
        integral = 0.;
        ctimer(&t1, &tucpu, &tscpu);
        for (i = 0; i < N; i++) {
            x = xStart + xInterval * ((double)i + 0.5);
            integral += polynom(f, M, x) * xInterval;
        }
        integral += C;
        ctimer(&t2, &tucpu, &tscpu);
        // Sequential results display
        printf("Sequential integral =\t%.3f\n", integral);
        printf("Time:\t%.9f seconds\n", (float)(t2 - t1));
    }

    // Data broadcast to other processes
    ctimer(&t1, &tucpu, &tscpu);
    MPI_Bcast(f, M + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Local computation in each processor
    mpiIntegral = 0.;
    for (i = 0; i < numBlocks; i++) {
        x = xStart + xInterval * ((double)(myRank * numBlocks + i) + 0.5);
        mpiIntegral += polynom(f, M, x) * xInterval;
    }

    // Result gathering in the root processor
    MPI_Reduce(&mpiIntegral, &integral, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    integral += C;
    ctimer(&t2, &tucpu, &tscpu);

    // Results display
    if (!myRank) {
        printf("Parallel integral =\t%.3f\n", integral);
        printf("Time:\t%.9f seconds\n", (float)(t2 - t1));
    }

    MPI_Finalize();
    return 0;
}

double polynom(double *f, int grade, double x) {
    double fx = 0.;
    int n;
    for (n = 0; n <= grade; n++) {
        fx += f[n] * pow(x, n);
    }
    return fx;
}
