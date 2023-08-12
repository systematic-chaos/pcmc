#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
    
    // Get the number of processes
    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    
    // Get the rank of the process
    int worldRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    
    // Get the name of the processor
    char processorName[MPI_MAX_PROCESSOR_NAME];
    int nameLen;
    MPI_Get_processor_name(processorName, &nameLen);
    
    // Print off a hello world message
    printf("Hello world from processor %s, rank %d out of %d processors\n",
        processorName, worldRank + 1, worldSize);
    
    // Finalize the MPI environment
    MPI_Finalize();
}

