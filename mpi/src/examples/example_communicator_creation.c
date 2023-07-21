#include <stdlib.h>
#include <math.h>
#include <mpi.h>

void rowCommunicator(MPI_Comm* firstRowComm);

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    MPI_Comm firstRowComm;
    rowCommunicator(&firstRowComm);

    MPI_Finalize();
    return 0;
}

void rowCommunicator(MPI_Comm* firstRowComm) {
    MPI_Group MPI_GROUP_WORLD;
    MPI_Group firstRowGroup;
    int *groupRanges;
    int p, q, proc;

    // Compute the number of processes, p, and q = log2(p)
    MPI_Comm_size(MPI_COMM_WORLD, &proc);
    q = (int)log2(proc);

    // Obtain processes for the new communicator
    groupRanges = (int*)malloc(q * sizeof(int));
    for (p = 0; p < q; p++) {
        groupRanges[p] = p;
    }

    // Obtain the group contained in MPI_COMM_WORLD
    MPI_Comm_group(MPI_COMM_WORLD, &MPI_GROUP_WORLD);

    // Create a new group
    MPI_Group_incl(MPI_GROUP_WORLD, q, groupRanges, &firstRowGroup);

    // Create a new communicator
    MPI_Comm_create(MPI_COMM_WORLD, firstRowGroup, firstRowComm);

    MPI_Comm rowComm;
    int row, range = proc;

    // range in MPI_COMM_WORLD
    row = range / q;
    MPI_Comm_split(MPI_COMM_WORLD, row, range, &rowComm);
}
