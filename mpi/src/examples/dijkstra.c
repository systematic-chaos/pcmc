#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <mpi.h>
#include "ctimer.h"

#define N 128
#define INFINITE USHRT_MAX

void singleSource(int source, int *weights, int *lengths, int dim, MPI_Comm comm);
void dijkstra(int *lengths, int *weights, int *marker,
    int nLocal, int firstVtx, int lastVtx, int dim, MPI_Comm comm);
int* generateIntArray(int length, int limit);

int main(int argc, char **argv) {
    int *weights, *lengths;
    weights = generateIntArray(N, 100);
    lengths = generateIntArray(N * N, 200);

    MPI_Init(&argc, &argv);

    singleSource(0, weights, lengths, N, MPI_COMM_WORLD);

    MPI_Finalize();

    free(weights);
    free(lengths);
}

void singleSource(int source, int *weights, int *lengths, int dim, MPI_Comm comm) {
    int nLocal;     // The number of vertices stored locally
    int *marker;    // Used to mark the vertices belonging to V_0
    int firstVtx;   // The index number of the first vertex that is stored locally
    int lastVtx;    // The index number of the last vertex that is stored locally
    int i;
    int np, myRank;
    double t1, t2, tucpu, tscpu;
    MPI_Status status;

    MPI_Comm_size(comm, &np);
    MPI_Comm_rank(comm, &myRank);

    nLocal = dim / np;
    firstVtx = myRank * nLocal;
    lastVtx = firstVtx + nLocal - 1;

    // Set the initial distances from source to all other vertices
    for (i = 0; i < nLocal; i++) {
        lengths[i] = weights[source * nLocal + i];
    }

    // This array is used to indicate whether the shortest path to a vertex has been found.
    // If `marker[v]` is one, then the shortest path to `v` has been found, otherwise it has not.
    marker = malloc(nLocal * sizeof(int));
    for (i = 0; i < nLocal; i++) {
        marker[i] = 1;
    }

    // The process that stores the source vertex, mark it as being seen
    if (source >= firstVtx && source <= lastVtx) {
        marker[source - firstVtx] = 0;
    }

    ctimer(&t1, &tucpu, &tscpu);
    dijkstra(lengths, weights, marker, nLocal, firstVtx, lastVtx, dim, comm);
    ctimer(&t2, &tucpu, &tscpu);

    if (!myRank) {
        printf("Time:\t%.9f seconds\n", (float)(t2 - t1));
    }

    free(marker);
}

// The main loop of Dijkstra's algorithm
void dijkstra(int *lengths, int *weights, int *marker,
    int nLocal, int firstVtx, int lastVtx, int dim, MPI_Comm comm) {
    int u, uDist;
    int lMinPair[2], gMinPair[2];

    int i, j;
    for (i = 1; i < dim; i++) {
        // Step 1: Find the local vertex that is at the smallest distance from source
        lMinPair[0] = INFINITE;
        lMinPair[1] = -1;
        for (j = 0; j < nLocal; j++) {
            if (marker[j] && lengths[j] < lMinPair[0]) {
                lMinPair[0] = lengths[j];
                lMinPair[1] = firstVtx + j;
            }
        }

        // Step 2: Compute the global minimum vertex and insert it into V_c
        MPI_Allreduce(lMinPair, gMinPair, 1, MPI_2INT, MPI_MINLOC, comm);
        uDist = gMinPair[0];
        u = gMinPair[1];

        // The process that stores the minimum vertex, mark it as being seen
        if (u == lMinPair[1]) {
            marker[u - firstVtx] = 0;
        }

        // Step 3: Update the distances given that `u` got inserted
        for (j = 0; j < nLocal; j++) {
            if (marker[j] && uDist + weights[u * nLocal + j] < lengths[j]) {
                lengths[j] = uDist + weights[u * nLocal + j];
            }
        }
    }
}

// Allocate memory and set the initial distance from a node to all other vertices
int* generateIntArray(int length, int limit) {
    int *arrPtr;
    arrPtr = (int*)malloc(length * sizeof(int));
    
    srand(time(NULL));
    
    int i;
    for (i = 0; i < length; i++) {
        arrPtr[i] = rand() % limit + limit;
    }

    return arrPtr;
}
