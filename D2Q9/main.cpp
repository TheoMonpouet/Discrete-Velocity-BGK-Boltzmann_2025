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


// check_nan(): Check if ghat contains NaN, indicating instability in the solution
// Input:
//    valarray<complex<double>>& ghat: Particle velocity probability distribution
// Output:
//    bool has_nan: True if ghat contains NaN, false otherwise 
bool check_nan(valarray<complex<double>>& ghat) {
    bool has_nan = false;

    for (int m = 0; m < 9; m++) {
        int lattice_offset = m * Constants::local_alloc_fs;
        
        for (int i = 0; i < Constants::local_N_fs; i++) {
            for (int j = 0; j < Constants::N; j++) {
                int index = lattice_offset + i * Constants::N + j;

                if (isnan(ghat[i].real()) || isnan(ghat[i].imag())) {
                    has_nan = true;
                    break;
                }
            }
        }
    }
    return has_nan;
}




// Main function
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

    // Save initial condition
    double T = Constants::T0;
    save_to_file_routine(ghat, Constants::T0, 0);


    // Main time loop
    if (rank == 0) cout << "Starting..." << endl;

    for (int ti = 1; ti < Constants::Nd + 1; ti++) {
        // Stepping in time and updating ghat
        RK4stepping(ghat);
        T += Constants::dt;

        // To get regular updates
        if (ti % 1000 == 0 && rank == 0) cout << T << endl;

        // Only runs depending on Nsave value
        if (ti % Constants::TSCREEN == 1) {
            // Check if solution is stable
            if (check_nan(ghat)) {
                cerr << "has NaN!" << endl;
                if (rank == 0) MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }

            // Save solution to file
            save_to_file_routine(ghat, T, ti);
        }
    }

    
    return 0;
}
