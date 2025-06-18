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

    // Initializing MPI and FFTW
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


    // Set initial condition
    valarray<complex<double>> ghat(Constants::local_alloc_fs * 9);
    if (Constants::init_condition == "fromfile") set_initial_ghat_from_file(ghat);
    else set_initial_ghat_from_closed_form(ghat);


    // Main time loop
    if (rank == 0) cout << "Starting..." << endl;
    double T = Constants::T0;

    



    return 0;
}

//TODO change 9 to lattice number