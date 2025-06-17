"""
    FFTHandler.cpp: struct with static members to handle and execute the fast Fourier transform and the inverse fast Fourier transform.
   
    Author: Théo Monpouet Ekeram
    As part of a master thesis in Computational Mathematics, KTH Royal Institute of Technology
    Thesis name: 'Parallel implementation and analysis of the discrete-velocity BGK Boltzmann method'
    Date: 2025/06/01
"""


#include <iostream>
#include <math.h>
#include <complex>
#include <valarray>
#include <algorithm>

#include <mpi.h>
#include <fftw3-mpi.h>


using namespace std;

struct FFTHandler {
    static fftw_plan forward_plan;
    static fftw_plan inverse_plan;

    static valarray<double> real_data;
    static valarray<complex<double>> complex_data;


    // initialize(): Static method to initialize data sizes and fftw plans
    static void initialize() {
        real_data.resize(2 * Constants::local_alloc_ps);
        complex_data.resize(Constants::local_alloc_fs);

        // Initializing forward and inverse plans with the flag for leaving out last transform
        forward_plan = fftw_mpi_plan_dft_r2c_2d(Constants::N, Constants::N, &real_data[0], reinterpret_cast<fftw_complex*>(&complex_data[0]), MPI_COMM_WORLD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT);
        inverse_plan = fftw_mpi_plan_dft_c2r_2d(Constants::N, Constants::N, reinterpret_cast<fftw_complex*>(&complex_data[0]), &real_data[0], MPI_COMM_WORLD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN);
    }

    
    // cleanup(): Static method to free memory
    static void cleanup() {
        fftw_destroy_plan(forward_plan);
        fftw_destroy_plan(inverse_plan);
    }


    // execute_fwd(): Static method to execute the fft.
    // Inputs:
    //    - valarray<double>& input: A valarray containing doubles to be used as input
    //    - valarray<complex<double>>& output: A valarray containing complex<double> to be overridden as output
    static void execute_fwd(valarray<double>& input, valarray<complex<double>>& output) {

        // Copy input vector to struct member
        for (int i = 0; i < Constants::local_N_ps; i++) {
            for (int j = 0; j < Constants::N; j++) {
                int index = i * 2*(Constants::N/2+1) + j;
                real_data[index] = input[index];
            }
        }

        // Run parallel fft routine
        fftw_execute(forward_plan);

        // Override output vector with the result of the fft
        for (int i = 0; i < Constants::local_alloc_fs; i++) {
            output[i] = complex_data[i];
        }
    }


    // execute_bkwd(): Static method to execute the inverse fft.
    // Inputs:
    //    - valarray<complex<double>>& input: A valarray containing complex<double> to be used as input
    //    - valarray<double>& output: A valarray containing doubles to be overridden as output
    static void execute_bkwd(valarray<complex<double>>& input, valarray<double>& output) {

        // Copy input vector to struc member
        for (int i = 0; i < Constants::local_alloc_fs; i++) {
            complex_data[i] = input[i];
        }

        // Run parallel inverse fft routine
        fftw_execute(inverse_plan);

        // Override output vector with the normalized result of the inverse fft
        for (int i = 0; i < Constants::local_N_ps; i++) {
            for (int j = 0; j < Constants::N; j++) {
                int index = i * 2*(Constants::N/2+1) + j;
                output[index] = real_data[index] / (Constants::N * Constants::N);
            }
        }
    }

    
    // execute_bkwd(): Static method to execute the inverse fft for multiple inputs in a big array.
    // Inputs:
    //    - valarray<complex<double>>& input: A valarray containing complex<double> to be used as input
    //    - valarray<double>& output: A valarray containing doubles to be overridden as output
    //    - int offset_index: An integer representing which part of the input array to operate on
    static void execute_bkwd_index(const valarray<complex<double>>& input, valarray<double>& output, int offset_index) {

        // Copy the part of the input vector indicated by offset_index to struct member
        for (int i = 0; i < Constants::local_alloc_fs; i++) {
            complex_data[i] = input[i + Constants::local_alloc_fs * offset_index];
        }

        // Run parallel fft routine
        fftw_execute(inverse_plan);

        // Override the part of the output vector indicated by offset_index with the normalized result
        for (int i = 0; i < Constants::local_N_ps; i++) {
            for (int j = 0; j < Constants::N; j++) {
                int index = i * 2*(Constants::N/2+1) + j;
                output[index + 2*Constants::local_alloc_ps * offset_index] = real_data[index] / (Constants::N * Constants::N);
            }
        }
    }
};

// Struct members
fftw_plan FFTHandler::forward_plan;
fftw_plan FFTHandler::inverse_plan;
valarray<double> FFTHandler::real_data;
valarray<complex<double>> FFTHandler::complex_data;

