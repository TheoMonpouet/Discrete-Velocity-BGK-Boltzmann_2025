/*
    init_variables.cpp: struct with static members to compute and store all constant variables used in the program.
    Change physical and numerical parameters here.
   
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
#include <filesystem> // For creating folder


#include <mpi.h>
#include <fftw3-mpi.h>



using namespace std;

// Custom mod function
int mod(int a, int b) {
    return (a % b + b) % b; // -6%12 would be -6 using c++:s mod, "should" be +6
}


struct Constants {
    // Adjustable physical parameters
    static const double nu;
    static const double epsilon;
    static const double Lx;
    static const double Ly;
    static const double T0;
    static const double T1;

    // Adjustable Numerical parameters
    static const int N;
    static const double dt;

    // Initial conditions
    static const string init_condition;
    static const string init_file_path;

    // Saving solution
    static const string result_file_path;
    static const string saving_sol;
    static const int Nsave;

    // Fixed parameters
    static const int N_half;
    static const double c_s;
    static const double dx;
    static const double dy;
    static const int Nd;
    static const int TSCREEN;

    // MPI constants
    static ptrdiff_t local_alloc_ps;
    static ptrdiff_t local_alloc_fs;
    static ptrdiff_t local_N;
    static ptrdiff_t local_start;

    static int rank;
    static int size;

    // Wavenumbers
    static valarray<complex<double>> Kx;
    static valarray<complex<double>> Ky;
    static valarray<complex<double>> Kz;


    // Operators
    static valarray<complex<double>> Lap_hat;
    static valarray<complex<double>> Pois_hat;
    static valarray<complex<double>> FTh;

    // dealias filter
    static valarray<complex<double>> dealias;

    // Weights
    static int lattice_number;
    static valarray<double> weights;



    
    // set_sizes(): Static method to compute and set sizes of operator arrays
    // Inputs:
    //    - int size_: How many total processes there are.
    //    - int rank_: Which process (0-size_) is the current running process.
    static void set_sizes(int size_, int rank_) {
        // Load distribution between processes
        local_alloc_fs = fftw_mpi_local_size_3d(N, N, N/2+1, MPI_COMM_WORLD, &local_N, &local_start);
        local_alloc_ps = 2 * local_alloc_fs;

        size = size_;
        rank = rank_;

        // initializing operator sizes
        Kx.resize(local_alloc_fs);
        Ky.resize(local_alloc_fs);
        Kz.resize(local_alloc_fs);
        Lap_hat.resize(local_alloc_fs);
        Pois_hat.resize(local_alloc_fs);
        dealias.resize(local_alloc_fs);
        
        FTh.resize(local_alloc_fs * lattice_number);
    }


    // set_operators(): Static method to compute and set operators Kx, Ky, Lap_hat, Pois_hat, FTh and dealias
    static void set_operators() {

        valarray<double> kmod(local_alloc_fs);

        for (int i = 0; i < local_N; i++) {
            int tempx = mod((int) ((local_start + i + 1) - ceil(N / 2.0 + 1)), N) - floor(N / 2.0);

            for (int j = 0; j < N; j++) {
                int tempy = mod((int) ((j + 1) - ceil(N / 2.0 + 1)), N) - floor(N / 2.0);

                for (int k = 0; k < N/2+1; k++) {
                    int tempz = mod((int) ((k + 1) - ceil(N / 2.0 + 1)), N) - floor(N / 2.0);

                    int index = (i*N + j) * (N/2+1) + k;
                    Kx[index] = complex<double>(0, (2*M_PI / Lx) * tempx);
                    Ky[index] = complex<double>(0, (2*M_PI / Lx) * tempy);
                    Kz[index] = complex<double>(0, (2*M_PI / Lx) * tempz);

                    // Lap_hat
                    double temp = real(Kx[index]*Kx[index] + Ky[index]*Ky[index] + Kz[index]*Kz[index]);
                    Lap_hat[index] = complex<double>(temp, 0);

                    // Pois_hat
                    Pois_hat[index] = complex<double>(temp, 0);
                    if (index == 0 && rank == 0) Pois_hat[index] = complex<double>(1.0, 0);

                    // kmod
                    kmod[index] = sqrt(abs(Kx[index])*abs(Kx[index]) + abs(Ky[index])*abs(Ky[index]) + abs(Kz[index])*abs(Kz[index]));


                    if (local_start + i == round(N/2)) Kx[index] *= 0;
                    if (j == round(N/2)) Ky[index] *= 0;
                    if (k == round(N/2)) Kz[index] *= 0;
                }
            }
        }
        

        // FTh
        int offset = Constants::local_alloc_fs;

        FTh[slice(0*offset, offset, 1)]  = -(1/(epsilon*epsilon*nu) + (Kx-Kx) / epsilon);    // 0,0,0
        FTh[slice(1*offset, offset, 1)]  = -(1/(epsilon*epsilon*nu) + (Kx) / epsilon);       // 1,0,0
        FTh[slice(2*offset, offset, 1)]  = -(1/(epsilon*epsilon*nu) + (-Kx) / epsilon);      // -1,0,0
        FTh[slice(3*offset, offset, 1)]  = -(1/(epsilon*epsilon*nu) + (Ky) / epsilon);       // 0,1,0
        FTh[slice(4*offset, offset, 1)]  = -(1/(epsilon*epsilon*nu) + (-Ky) / epsilon);      // 0,-1,0
        FTh[slice(5*offset, offset, 1)]  = -(1/(epsilon*epsilon*nu) + (Kz) / epsilon);       // 0,0,1
        FTh[slice(6*offset, offset, 1)]  = -(1/(epsilon*epsilon*nu) + (-Kz) / epsilon);      // 0,0,-1
        FTh[slice(7*offset, offset, 1)]  = -(1/(epsilon*epsilon*nu) + (Kx + Ky) / epsilon);  // 1,1,0
        FTh[slice(8*offset, offset, 1)]  = -(1/(epsilon*epsilon*nu) + (-Kx - Ky) / epsilon); // -1,-1,0
        FTh[slice(9*offset, offset, 1)]  = -(1/(epsilon*epsilon*nu) + (Kx + Kz) / epsilon);  // 1,0,1
        FTh[slice(10*offset, offset, 1)] = -(1/(epsilon*epsilon*nu) + (-Kx - Kz) / epsilon); // -1,0,-1
        FTh[slice(11*offset, offset, 1)] = -(1/(epsilon*epsilon*nu) + (Ky + Kz) / epsilon);  // 0,1,1
        FTh[slice(12*offset, offset, 1)] = -(1/(epsilon*epsilon*nu) + (-Ky - Kz) / epsilon); // 0,-1,-1
        FTh[slice(13*offset, offset, 1)] = -(1/(epsilon*epsilon*nu) + (Kx - Ky) / epsilon);  // 1,-1,0
        FTh[slice(14*offset, offset, 1)] = -(1/(epsilon*epsilon*nu) + (-Kx + Ky) / epsilon); // -1,1,0
        FTh[slice(15*offset, offset, 1)] = -(1/(epsilon*epsilon*nu) + (Kx - Kz) / epsilon);  // 1,0,-1
        FTh[slice(16*offset, offset, 1)] = -(1/(epsilon*epsilon*nu) + (-Kx + Kz) / epsilon); // -1,0,1
        FTh[slice(17*offset, offset, 1)] = -(1/(epsilon*epsilon*nu) + (Ky - Kz) / epsilon);  // 0,1,-1
        FTh[slice(18*offset, offset, 1)] = -(1/(epsilon*epsilon*nu) + (-Ky + Kz) / epsilon); // 0,-1,1
        

        // kcut
        double local_max = kmod.max();
        double kcut;
        MPI_Allreduce(&local_max, &kcut, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        kcut *= 2.0/3.0;
        

        // dealias
        for (int i = 0; i < local_N; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N/2+1; k++) {
                    int index = (i*N + j) * (N/2+1) + k;
                    dealias[index] = complex<double>(exp(-36.0 * pow(kmod[index]/kcut, 36)), 0);
                }
            }
        }
    }



    static string to_string_rounded(double value, int decimals = 2) {
        ostringstream out;
        out << std::fixed << std::setprecision(decimals) << value;
        return out.str();
    }


    // initialize(): Static method to initialize sizes and operators
    static void initialize(int size_, int rank_) {
        set_sizes(size_, rank_);
        set_operators();
    }
};


// Adjustable physical parameters
const double Constants::nu = pow(10, -4);
const double Constants::epsilon = 1;
const double Constants::Lx = 1.0;
const double Constants::Ly = 1.0;
const double Constants::T0 = 0.0;
const double Constants::T1 = 1;

// Adjustable Numerical parameters
const int Constants::N = 64;
const double Constants::dt = pow(10, -4);


// Initial condition
// "c1":  Closed form 1
// TODO: Fromfile
const string Constants::init_condition = "c1";
const string Constants::init_file_path = "/"; // File path of where to read initial condition


// Saving solution
const string Constants::result_file_path = "/"; // File path (folder) of where to save solution (end with "/")
const string Constants::saving_sol = "we"; // "w": Save vorticity, "e": save error, "we": save both (only applicable to IC "tg", for "ptg" only vorticity gets saved)
const int Constants::Nsave = 1000; // Number of steps saved


// Fixed parameters
const int Constants::N_half = Constants::N/2 + 1;
const double Constants::c_s = pow(1.0/3.0, 1.0/2.0);
const double Constants::dx = Lx/N;
const double Constants::dy = Ly/N;
const int Constants::Nd = round(T1/dt);
const int Constants::TSCREEN = floor(T1 / (dt * Nsave));


// MPI constants
ptrdiff_t Constants::local_alloc_ps;
ptrdiff_t Constants::local_alloc_fs;
ptrdiff_t Constants::local_N;
ptrdiff_t Constants::local_start;

int Constants::rank;
int Constants::size;

// Wavenumbers
valarray<complex<double>> Constants::Kx;
valarray<complex<double>> Constants::Ky;
valarray<complex<double>> Constants::Kz;


// Operators
valarray<complex<double>> Constants::Lap_hat;
valarray<complex<double>> Constants::Pois_hat;
valarray<complex<double>> Constants::FTh;


// Dealias filter
valarray<complex<double>> Constants::dealias;




// Weights
int Constants::lattice_number = 19;
valarray<double> Constants::weights{1.0/3.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0}; 
