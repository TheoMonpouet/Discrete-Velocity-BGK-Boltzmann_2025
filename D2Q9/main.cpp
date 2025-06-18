#include <iostream>
#include <math.h>
#include <complex>
#include <valarray>
#include <algorithm>  // For std::max_element
#include <chrono>
#include <mpi.h>
#include <fftw3-mpi.h>

#include "init_variables.cpp"
#include "FFTHandler.cpp"
#include "temp_variables.cpp"
#include "functions.cpp"



int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);
    fftw_mpi_init();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize contants
    if (rank == 0) cout << "Init..." << endl;
    Constants::initialize(size, rank);
    Temp::initialize();
    if (rank == 0){
        cout << "epsilon: " << Constants::epsilon << endl;
        cout << "dt: " << Constants::dt << endl;
        cout << "Nd: " << Constants::Nd << endl;
        cout << "# Processes: " << size << endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    

    // Initialize FFT routines
    FFTHandler::initialize();





    return 0;
}