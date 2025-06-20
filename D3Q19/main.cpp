/*
    main.cpp: The main program, including a check for solution stability and the main time loop.
   
    Author: Théo Monpouet Ekeram
    As part of a master thesis in Computational Mathematics, KTH Royal Institute of Technology
    Thesis name: 'Parallel implementation and analysis of the discrete-velocity BGK Boltzmann method'
    Date: 2025/06/01
*/


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


using namespace std;


// Main function
int main(int argc, char **argv) {

    // Initializing MPI and FFTW
    MPI_Init(&argc, &argv);
    fftw_mpi_init();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);



    // Initialize structs
    Constants::initialize(size, rank);
    Temp::initialize();
    FFTHandler::initialize();
    if (rank == 0){
        cout << "epsilon: " << Constants::epsilon << endl;
        cout << "dt: " << Constants::dt << endl;
        cout << "Nd: " << Constants::Nd << endl;
        cout << "# Processes: " << size << endl;
    }

    // Set starting ghat
    valarray<complex<double>> ghat(Constants::local_alloc_fs * Constants::lattice_number);
    set_initial_ghat(ghat);
    



    // Main time loop
    for (int ti = 1; ti < Constants::Nd + 1; ti++) {
        RK4stepping(ghat);

        // To get regular updates
        if (ti % 1000 == 0 && rank == 0) cout << ti << endl;

        // Only runs depending on Nsave value
        if (ti % Constants::TSCREEN == 1) save_to_file_routine(ghat, ti);
    }
    



    return 0;
}