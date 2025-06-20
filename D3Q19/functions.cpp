/*
    functions.cpp: the functions used in the program for the timestepping and saving of solutions.
   
    Author: Théo Monpouet Ekeram
    As part of a master thesis in Computational Mathematics, KTH Royal Institute of Technology
    Thesis name: 'Parallel implementation and analysis of the discrete-velocity BGK Boltzmann method'
    Date: 2025/06/01
*/


#include <iostream>
#include <math.h>
#include <complex>
#include <valarray>
#include <algorithm>
#include <fstream>
#include <iomanip>

#include <mpi.h>
#include <fftw3-mpi.h>

using namespace std;




// u2geq_hat(): function that uses the velocity components, density, lattice weights and Knudsen number
//              to compute the particle velocity distribution g_eq in Fourier space. Using dealiasing for 
//              non-linear terms.
// Inputs:
//       - valarray<complex<double>>& geqh: array of complex<double> to be overwritten with the output of the function
void u2geq_hat(valarray<complex<double>>& geqh) {   

    // Coefficients with the speed of sound constant
    double C2 = 1.0 / (Constants::c_s* Constants::c_s);
    double C4 = 0.5 * C2 * C2;

    // Transform velocity to physical space
    FFTHandler::execute_bkwd(Temp::u1h, Temp::u1);
    FFTHandler::execute_bkwd(Temp::u2h, Temp::u2);
    FFTHandler::execute_bkwd(Temp::u3h, Temp::u3);

    // Pre-compute products in physical space
    for (int i  = 0; i < Constants::local_alloc_ps; i++) {
        Temp::u11[i] = Temp::u1[i] * Temp::u1[i];
        Temp::u22[i] = Temp::u2[i] * Temp::u2[i];
        Temp::u33[i] = Temp::u3[i] * Temp::u3[i];

        Temp::u12[i] = Temp::u1[i] * Temp::u2[i];
        Temp::u13[i] = Temp::u1[i] * Temp::u3[i];
        Temp::u23[i] = Temp::u2[i] * Temp::u3[i];

        Temp::uabs[i] = (C2/2.0) * (Temp::u11[i] + Temp::u22[i] + Temp::u33[i]);
    }

    // Transform the products to Fourier space
    FFTHandler::execute_fwd(Temp::u11, Temp::u11h);
    FFTHandler::execute_fwd(Temp::u22, Temp::u22h);
    FFTHandler::execute_fwd(Temp::u33, Temp::u33h);

    FFTHandler::execute_fwd(Temp::u12, Temp::u12h);
    FFTHandler::execute_fwd(Temp::u13, Temp::u13h);
    FFTHandler::execute_fwd(Temp::u23, Temp::u23h);

    FFTHandler::execute_fwd(Temp::uabs, Temp::uabsh);


    // Dealias non-linear terms
    for (int i  = 0; i < Constants::local_alloc_fs; i++) {
        Temp::u11h[i] = Constants::epsilon * Temp::u11h[i] * Constants::dealias[i];
        Temp::u22h[i] = Constants::epsilon * Temp::u22h[i] * Constants::dealias[i];
        Temp::u33h[i] = Constants::epsilon * Temp::u33h[i] * Constants::dealias[i];

        Temp::u12h[i] = Constants::epsilon * Temp::u12h[i] * Constants::dealias[i];
        Temp::u13h[i] = Constants::epsilon * Temp::u13h[i] * Constants::dealias[i];
        Temp::u23h[i] = Constants::epsilon * Temp::u23h[i] * Constants::dealias[i];

        Temp::uabsh[i]= Constants::epsilon * Temp::uabsh[i]* Constants::dealias[i];
        Temp::rhoh[i] = Constants::epsilon * Temp::rhoh[i] * Constants::dealias[i];
    }


    // Compute geqh from velocity and density fields using the weights
    int offset = Constants::local_alloc_fs;
    for (int i = 0; i < Constants::local_alloc_fs; i++) {
        geqh[i + 0*offset]  = (Temp::rhoh[i]                                     - Temp::uabsh[i]                                                           ) * Constants::weights[0];

        geqh[i + 1*offset]  = (Temp::rhoh[i] + C2*Temp::u1h[i]                   - Temp::uabsh[i] + C4*Temp::u11h[i]                                        ) * Constants::weights[1];
        geqh[i + 2*offset]  = (Temp::rhoh[i] - C2*Temp::u1h[i]                   - Temp::uabsh[i] + C4*Temp::u11h[i]                                        ) * Constants::weights[2];
        geqh[i + 3*offset]  = (Temp::rhoh[i] + C2*Temp::u2h[i]                   - Temp::uabsh[i] + C4*Temp::u22h[i]                                        ) * Constants::weights[3];
        geqh[i + 4*offset]  = (Temp::rhoh[i] - C2*Temp::u2h[i]                   - Temp::uabsh[i] + C4*Temp::u22h[i]                                        ) * Constants::weights[4];
        geqh[i + 5*offset]  = (Temp::rhoh[i] + C2*Temp::u3h[i]                   - Temp::uabsh[i] + C4*Temp::u33h[i]                                        ) * Constants::weights[5];
        geqh[i + 6*offset]  = (Temp::rhoh[i] - C2*Temp::u3h[i]                   - Temp::uabsh[i] + C4*Temp::u33h[i]                                        ) * Constants::weights[6];
        
        geqh[i + 7*offset]  = (Temp::rhoh[i] + C2*( Temp::u1h[i] + Temp::u2h[i]) - Temp::uabsh[i] + C4*2*Temp::u12h[i] + C4*Temp::u11h[i] + C4*Temp::u22h[i]) * Constants::weights[7];
        geqh[i + 8*offset]  = (Temp::rhoh[i] + C2*(-Temp::u1h[i] - Temp::u2h[i]) - Temp::uabsh[i] + C4*2*Temp::u12h[i] + C4*Temp::u11h[i] + C4*Temp::u22h[i]) * Constants::weights[8];
        geqh[i + 9*offset]  = (Temp::rhoh[i] + C2*( Temp::u1h[i] + Temp::u3h[i]) - Temp::uabsh[i] + C4*2*Temp::u13h[i] + C4*Temp::u11h[i] + C4*Temp::u33h[i]) * Constants::weights[9];
        geqh[i + 10*offset] = (Temp::rhoh[i] + C2*(-Temp::u1h[i] - Temp::u3h[i]) - Temp::uabsh[i] + C4*2*Temp::u13h[i] + C4*Temp::u11h[i] + C4*Temp::u33h[i]) * Constants::weights[10];
        geqh[i + 11*offset] = (Temp::rhoh[i] + C2*( Temp::u2h[i] + Temp::u3h[i]) - Temp::uabsh[i] + C4*2*Temp::u23h[i] + C4*Temp::u22h[i] + C4*Temp::u33h[i]) * Constants::weights[11];
        geqh[i + 12*offset] = (Temp::rhoh[i] + C2*(-Temp::u2h[i] - Temp::u3h[i]) - Temp::uabsh[i] + C4*2*Temp::u23h[i] + C4*Temp::u22h[i] + C4*Temp::u33h[i]) * Constants::weights[12];
        
        geqh[i + 13*offset] = (Temp::rhoh[i] + C2*( Temp::u1h[i] - Temp::u2h[i]) - Temp::uabsh[i] - C4*2*Temp::u12h[i] + C4*Temp::u11h[i] + C4*Temp::u22h[i]) * Constants::weights[13];
        geqh[i + 14*offset] = (Temp::rhoh[i] + C2*(-Temp::u1h[i] + Temp::u2h[i]) - Temp::uabsh[i] - C4*2*Temp::u12h[i] + C4*Temp::u11h[i] + C4*Temp::u22h[i]) * Constants::weights[14];
        geqh[i + 15*offset] = (Temp::rhoh[i] + C2*( Temp::u1h[i] - Temp::u3h[i]) - Temp::uabsh[i] - C4*2*Temp::u13h[i] + C4*Temp::u11h[i] + C4*Temp::u33h[i]) * Constants::weights[15];
        geqh[i + 16*offset] = (Temp::rhoh[i] + C2*(-Temp::u1h[i] + Temp::u3h[i]) - Temp::uabsh[i] - C4*2*Temp::u13h[i] + C4*Temp::u11h[i] + C4*Temp::u33h[i]) * Constants::weights[16];
        geqh[i + 17*offset] = (Temp::rhoh[i] + C2*( Temp::u2h[i] - Temp::u3h[i]) - Temp::uabsh[i] - C4*2*Temp::u23h[i] + C4*Temp::u22h[i] + C4*Temp::u33h[i]) * Constants::weights[17];
        geqh[i + 18*offset] = (Temp::rhoh[i] + C2*(-Temp::u2h[i] + Temp::u3h[i]) - Temp::uabsh[i] - C4*2*Temp::u23h[i] + C4*Temp::u22h[i] + C4*Temp::u33h[i]) * Constants::weights[18];
    }



}


// calc_rho(): Function to calculate the density field from the probability distribution functions
void calc_rho() {
    Temp::rho *= 0;

    for (int m = 0; m < Constants::lattice_number; m++) {

        for (int i = 0; i < Constants::local_N; i++) {
            for (int j = 0; j < Constants::N; j++) {
                for (int k = 0; k < Constants::N; k++) {
                    int index = (i*Constants::N + j) * (2*(Constants::N/2+1)) + k;
                    Temp::rho[index] += Temp::g[index + Constants::local_alloc_ps * m];
                }
            }
        }
    }
}



// rhou1u2u3(): function that computes the density field and reconstructs the velocity field in Fourier space from ghat
// Inputs:
//       - valarray<complex<double>>& ghat: Particle velocity probability distribution
void rhou1u2u3(const valarray<complex<double>>& ghat) {
    // Compute IFFT
    for (int i = 0; i < Constants::lattice_number; i++) {
        FFTHandler::execute_bkwd_index(ghat, Temp::g, i);
    }

    // Summing g to get density, stored in Temp::rho
    calc_rho();
    FFTHandler::execute_fwd(Temp::rho, Temp::rhoh);

    
    // NS velocity components
    int offset = Constants::local_alloc_fs;
    for (int i = 0; i < Constants::local_alloc_fs; i++) {
        Temp::u1h[i] = ghat[i + 1*offset]  - ghat[i + 2*offset]  + ghat[i + 7*offset]  - ghat[i + 8*offset]  + ghat[i + 9*offset] 
                     - ghat[i + 10*offset] + ghat[i + 13*offset] - ghat[i + 14*offset] + ghat[i + 15*offset] - ghat[i + 16*offset];
        Temp::u2h[i] = ghat[i + 3*offset]  - ghat[i + 4*offset]  + ghat[i + 7*offset]  - ghat[i + 8*offset]  + ghat[i + 11*offset] 
                     - ghat[i + 12*offset] - ghat[i + 13*offset] + ghat[i + 14*offset] + ghat[i + 17*offset] - ghat[i + 18*offset];
        Temp::u3h[i] = ghat[i + 5*offset]  - ghat[i + 6*offset]  + ghat[i + 9*offset]  - ghat[i + 10*offset] + ghat[i + 11*offset] 
                     - ghat[i + 12*offset] - ghat[i + 15*offset] + ghat[i + 16*offset] - ghat[i + 17*offset] + ghat[i + 18*offset];
    }


    // Leray projection for divergence-free solution
    for (int i = 0; i < Constants::local_alloc_fs; i++) {
        Temp::laP_hat[i] = Temp::u1h[i] * Constants::Kx[i] / Constants::Pois_hat[i] 
                         + Temp::u2h[i] * Constants::Ky[i] / Constants::Pois_hat[i]
                         + Temp::u3h[i] * Constants::Kz[i] / Constants::Pois_hat[i];
                         
        Temp::u1h[i] -= Constants::Kx[i] * Temp::laP_hat[i];
        Temp::u2h[i] -= Constants::Ky[i] * Temp::laP_hat[i];
        Temp::u3h[i] -= Constants::Kz[i] * Temp::laP_hat[i];
    }
}



// RHS(): Function to compute the right-hand-side in the RK4 timestepping formulation
// Inputs:
//      - valarray<complex<double>>& ghat: Particle velocity probability distribution
//      - valarray<complex<double>>& kn: The steps in the RK4 formulation, this variable is to be overwritten with the output of the function
void RHS(const valarray<complex<double>>& ghat, valarray<complex<double>>& kn) {
    // Computes u1h, u2h, u3h, rhoh
    rhou1u2u3(ghat);

    // Computes geqh
    u2geq_hat(Temp::geqh);

    // Computes RHS and overwrite result variable
    for (int i = 0; i < Constants::local_alloc_fs * Constants::lattice_number; i++) {
        kn[i] = Constants::dt * (Constants::FTh[i] * ghat[i] + Temp::geqh[i] / (Constants::epsilon * Constants::epsilon * Constants::nu));
    }
}



// RK4stepping(): Function to step forwards with the RK4 method
// Inputs:
//      - valarray<complex<double>>& ghat: Particle velocity probability distribution, to be updated with the result of the stepping
void RK4stepping(valarray<complex<double>>& ghat) {
    //k1
    for (int i = 0; i < Constants::local_alloc_fs * Constants::lattice_number; i++) {
        Temp::ghat_temp[i] = ghat[i];
    }
    RHS(Temp::ghat_temp, Temp::k1);

    //k2
    for (int i = 0; i < Constants::local_alloc_fs * Constants::lattice_number; i++) {
        Temp::ghat_temp[i] = ghat[i] + 0.5*Temp::k1[i];
    }
    RHS(Temp::ghat_temp, Temp::k2);

    //k3
    for (int i = 0; i < Constants::local_alloc_fs * Constants::lattice_number; i++) {
        Temp::ghat_temp[i] = ghat[i] + 0.5*Temp::k2[i];
    }
    RHS(Temp::ghat_temp, Temp::k3);

    //k4
    for (int i = 0; i < Constants::local_alloc_fs * Constants::lattice_number; i++) {
        Temp::ghat_temp[i] = ghat[i] + Temp::k3[i];
    }
    RHS(Temp::ghat_temp, Temp::k4);

    // Update ghat
    for (int i = 0; i < Constants::local_alloc_fs * Constants::lattice_number; i++) {
        ghat[i] += (Temp::k1[i] + 2.0*Temp::k2[i] + 2.0*Temp::k3[i] + Temp::k4[i]) / 6.0;
    }
}


// save_vel_to_file(): Function to save the velocity field to vtk file
// Inputs:
//      - valarray<double> u1: Array with the x-components of the velocity
//      - valarray<double> u2: Array with the y-components of the velocity
//      - valarray<double> u3: Array with the z-components of the velocity
//      - int ti: current time index
void save_vel_to_file(valarray<double> u1, valarray<double> u2, valarray<double> u3, int ti) {

    // Set file name
    ostringstream oss;
    oss << Constants::result_file_path << "/" << Constants::init_condition << "_eps" << Constants::epsilon << "_N" << Constants::N << "_dt" << Constants::dt << "_T" << Constants::T1 << "_nu" << Constants::nu << "_Nsave" << Constants::Nsave << "_rank" << Constants::rank << "_ti" << ti << ".vtk";
    ofstream file(oss.str());

    // VTK format headers
    file << "# vtk DataFile Version 3.0\n";
    file << "3D uniform grid with vectors\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";
    file << "DIMENSIONS " << Constants::N << " " << Constants::N << " " << Constants::N << "\n";
    file << "ORIGIN " << 0 << " " << 0 << " " << 0 << "\n";
    file << "SPACING " << Constants::dx << " " << Constants::dx << " " << Constants::dx << "\n";
    file << "POINT_DATA " << Constants::N*Constants::N*Constants::N << "\n";
    file << "VECTORS velocity float\n";

    // Save velocity field
    for (int i = 0; i < Constants::N; i++) {
        for (int j = 0; j < Constants::N; j++) {
            for (int k = 0; k < Constants::N; k++) {
                int index = (i*Constants::N + j) * (Constants::N/2+1) + k;
                
                file << u1[index] << " " << u2[index] << " " << u3[index] << "\n";
            }
        }
    }

    file.close();
    // cout << "VTK file written: " + file_name << endl;;

}


// save_err_to_file(): Function to save the error file
// Inputs:
//      - double u1_global_max: Value of the maximum error in the x-component
//      - double u2_global_max: Value of the maximum error in the y-component
//      - double u3_global_max: Value of the maximum error in the z-component
//      - int ti: current time index
void save_err_to_file(double u1_global_max, double u2_global_max, double u3_global_max) {
    string file_name = Constants::result_file_path + "/error.txt";
    ofstream out_error(file_name, std::ios::app);

    out_error << (1.0/3.0) / (u1_global_max + u2_global_max + u3_global_max) << "\n";
    out_error.close();
}


// save_to_file_routine(): Routine to save chosen info to file
// Inputs:
//      - valarray<complex<double>>& ghat: Particle velocity probability distribution
//      - int ti: current time index
void save_to_file_routine(const valarray<complex<double>>& ghat, int ti) {
    // Computing velocity from ghat
    rhou1u2u3(ghat);


    valarray<double> u1_local(Constants::local_alloc_ps);
    valarray<double> u2_local(Constants::local_alloc_ps);
    valarray<double> u3_local(Constants::local_alloc_ps);

    valarray<double> u1_local_filtered(Constants::local_N * Constants::N * Constants::N);
    valarray<double> u2_local_filtered(Constants::local_N * Constants::N * Constants::N);
    valarray<double> u3_local_filtered(Constants::local_N * Constants::N * Constants::N);

    valarray<double> u1_diff(Constants::local_N * Constants::N * Constants::N);
    valarray<double> u2_diff(Constants::local_N * Constants::N * Constants::N);
    valarray<double> u3_diff(Constants::local_N * Constants::N * Constants::N);

    valarray<double> u1_exact(Constants::local_N * Constants::N * Constants::N);
    valarray<double> u2_exact(Constants::local_N * Constants::N * Constants::N);
    valarray<double> u3_exact(Constants::local_N * Constants::N * Constants::N);

    // Transforming to physical space
    FFTHandler::execute_bkwd(Temp::u1h, u1_local);
    FFTHandler::execute_bkwd(Temp::u2h, u2_local);
    FFTHandler::execute_bkwd(Temp::u3h, u3_local);

    // Constants for computation of exact solution
    double wave = 2.0 * M_PI / Constants::Lx;
    double decay_factor = exp(-Constants::nu/3 * wave * wave * (ti * Constants::dt));
    
    // Computing exact solution and difference to simulation results
    for (int i = 0; i < Constants::local_N; i++) {
        double y = (i + Constants::local_start)*Constants::dx;
        for (int j = 0; j < Constants::N; j++) {
            double x = j*Constants::dx;
            for (int k = 0; k < Constants::N; k++) {
                double z = k*Constants::dx;

                int index = (i*Constants::N + j) * 2*(Constants::N/2+1) + k;
                int index_filtered = (i*Constants::N + j) * (Constants::N) + k;

                double x_prim = x + 0.25;
                double y_prim = y + 0.25;
                double z_prim = z + 0.25;

                
                u1_exact[index_filtered] = (sin(wave*z) + cos(wave*y))*decay_factor;
                u1_diff[index_filtered] = abs(u1_local[index] - u1_exact[index_filtered]);
               

                u2_exact[index_filtered] = (sin(wave*x) + cos(wave*z))*decay_factor;
                u2_diff[index_filtered] = abs(u2_local[index] - u2_exact[index_filtered]);

                u3_exact[index_filtered] = (sin(wave*y) + cos(wave*x))*decay_factor;
                u3_diff[index_filtered] = abs(u3_local[index] - u3_exact[index_filtered]);

                // Stability check
                if (isnan(u1_local[index]) || isnan(u2_local[index]) || isnan(u3_local[index])) {
                    cout << "has NaN!" << endl;
                    if (Constants::rank == 0) MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }

                u1_local_filtered[index_filtered] = u1_local[index];
                u2_local_filtered[index_filtered] = u2_local[index];
                u3_local_filtered[index_filtered] = u3_local[index];
            }
        }
    }


    // Computing Lmax error across all components
    double u1_local_max = u1_diff.max();
    double u1_global_max;

    double u2_local_max = u2_diff.max();
    double u2_global_max;

    double u3_local_max = u3_diff.max();
    double u3_global_max;

    // Reduce to find the maximum value across all processes
    MPI_Reduce(&u1_local_max, &u1_global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&u2_local_max, &u2_global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&u3_local_max, &u3_global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Saving velocity and/or error to file
    if (Constants::saving_sol == "we") {
        save_vel_to_file(u1_local_filtered, u2_local_filtered, u3_local_filtered, ti);
        if (Constants::rank == 0) save_err_to_file(u1_global_max, u2_global_max, u3_global_max);
    } else if (Constants::saving_sol == "w") {
        save_vel_to_file(u1_local_filtered, u2_local_filtered, u3_local_filtered, ti);
    } else if (Constants::saving_sol == "e") {
         if (Constants::rank == 0) save_err_to_file(u1_global_max, u2_global_max, u3_global_max);
    }
}



// set_initial_ghat(): Function to set the initial condition of ghat from the closed form expression
// Inputs:
//      - valarray<complex<double>>& ghat: Particle velocity probability distribution, to be overwritten with the initial condition of ghat
void set_initial_ghat(valarray<complex<double>>& ghat) {

    // Define Temp::u1, u2, u3
    for (int i = 0; i < Constants::local_N; i++) {
        double y = (i + Constants::local_start)*Constants::dx;

        for (int j = 0; j < Constants::N; j++) {
            double x = j*Constants::dx;
            for (int k = 0; k < Constants::N; k++) {
                double z = k*Constants::dx;

                int index = (i*Constants::N + j) * (2*(Constants::N/2+1)) + k;

                double factor = (4*sqrt(2)) / (3*sqrt(3));
                double a = M_PI / 6.0;
                double b = 5.0 * a;
                double wave = 2.0 * M_PI / Constants::Lx;

                Temp::u1[index] = sin(wave*z) + cos(wave*y);
                Temp::u2[index] = sin(wave*x) + cos(wave*z);
                Temp::u3[index] = sin(wave*y) + cos(wave*x);
            }
        }
    }

    
    // Compute Temp::u1h, u2h, u3h
    FFTHandler::execute_fwd(Temp::u1, Temp::u1h);
    FFTHandler::execute_fwd(Temp::u2, Temp::u2h);
    FFTHandler::execute_fwd(Temp::u3, Temp::u3h);


    // Define rho
    valarray<double> rho(1.0 / (Constants::N * Constants::N * Constants::N), Constants::local_alloc_ps);
    Temp::rho = rho;

    // Make Temp::rhoh
    FFTHandler::execute_fwd(Temp::rho, Temp::rhoh);

    // Set initial ghat
    u2geq_hat(ghat);
}





