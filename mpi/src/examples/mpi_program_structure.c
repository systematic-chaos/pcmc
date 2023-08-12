#include <mpi.h>

int main(int argc, char **argv) {   // main

    //Before here there is not any MPI function
    MPI_Init(&argc, &argv);

    // After here there is not any MPI function
    MPI_Finalize();

    return 0;
}
