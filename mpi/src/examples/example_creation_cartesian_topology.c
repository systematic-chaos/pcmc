#include <string.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    MPI_Comm cartComm, rowComm, colComm;
    int dimensions[2], coordinates[2], myCartRank;
    int periods[2], varyingCoords[2], reorder = 1;

    // Compute the number of processors, p, and q = sqrt(p)
    int p, q;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    q = (int)sqrt(p);
    memset(dimensions, 0, 2 * sizeof(int));

    dimensions[0] = dimensions[1] = q;

    // Retrieve the communicator bound to the (cartesian) topology
    MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, periods, reorder, &cartComm);

    // Compute coordinates and range within the topology
    MPI_Comm_rank(cartComm, &myCartRank);
    MPI_Cart_coords(cartComm, myCartRank, 2, coordinates);

    // Match rows and columns
    varyingCoords[0] = 0;
    varyingCoords[1] = 1;

    // Obtain the communicator bound to the processes on the same row
    MPI_Cart_sub(cartComm, varyingCoords, &rowComm);

    // Obtain the communicator bound to the processes on the same column
    varyingCoords[0] = 1;
    varyingCoords[1] = 0;
    MPI_Cart_sub(cartComm, varyingCoords, &colComm);

    MPI_Finalize();
    return 0;
}
