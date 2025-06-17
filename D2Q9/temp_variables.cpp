"""
    temp_variables.cpp: struct with static members to store all the temporary variables used in this program.
    This allows reusabilty instead of allocating new memory at each iterations
   
    Author: Théo Monpouet Ekeram
    As part of a master thesis in Computational Mathematics, KTH Royal Institute of Technology
    Thesis name: 'Parallel implementation and analysis of the discrete-velocity BGK Boltzmann method'
    Date: 2025/06/01
"""



#include <complex>
#include <valarray>
using namespace std;

struct Temp {
    static valarray<double> g;

    static valarray<double> u1;
    static valarray<double> u2;
    static valarray<double> u11;
    static valarray<double> u22;
    static valarray<double> u12;
    static valarray<double> uabs;
    static valarray<double> rho;

    static valarray<complex<double>> geqh;
    static valarray<complex<double>> u1h;
    static valarray<complex<double>> u2h;
    static valarray<complex<double>> u11h;
    static valarray<complex<double>> u12h;
    static valarray<complex<double>> u22h;
    static valarray<complex<double>> uabsh;
    static valarray<complex<double>> rhoh;
    static valarray<complex<double>> laP_hat;

    static valarray<complex<double>> k1;
    static valarray<complex<double>> k2;
    static valarray<complex<double>> k3;
    static valarray<complex<double>> k4;



    // initialize(): Setting the sizes of every member
    static void initialize() {
        g.resize(2*Constants::local_alloc_ps * 9);

        u1.resize(2*Constants::local_alloc_ps);
        u2.resize(2*Constants::local_alloc_ps);
        u11.resize(2*Constants::local_alloc_ps);
        u22.resize(2*Constants::local_alloc_ps);
        u12.resize(2*Constants::local_alloc_ps);
        uabs.resize(2*Constants::local_alloc_ps);
        rho.resize(2*Constants::local_alloc_ps);

        geqh.resize(Constants::local_alloc_fs * 9);
        u1h.resize(Constants::local_alloc_fs);
        u2h.resize(Constants::local_alloc_fs);
        u11h.resize(Constants::local_alloc_fs);
        u12h.resize(Constants::local_alloc_fs);
        u22h.resize(Constants::local_alloc_fs);
        uabsh.resize(Constants::local_alloc_fs);
        rhoh.resize(Constants::local_alloc_fs);
        laP_hat.resize(Constants::local_alloc_fs);

        k1.resize(Constants::local_alloc_fs * 9);
        k2.resize(Constants::local_alloc_fs * 9);
        k3.resize(Constants::local_alloc_fs * 9);
        k4.resize(Constants::local_alloc_fs * 9);

    }


};


valarray<double> Temp::g;

valarray<double> Temp::u1;
valarray<double> Temp::u2;
valarray<double> Temp::u11;
valarray<double> Temp::u22;
valarray<double> Temp::u12;
valarray<double> Temp::uabs;
valarray<double> Temp::rho;

valarray<complex<double>> Temp::geqh;
valarray<complex<double>> Temp::u1h;
valarray<complex<double>> Temp::u2h;
valarray<complex<double>> Temp::u11h;
valarray<complex<double>> Temp::u12h;
valarray<complex<double>> Temp::u22h;
valarray<complex<double>> Temp::uabsh;
valarray<complex<double>> Temp::rhoh;
valarray<complex<double>> Temp::laP_hat;


valarray<complex<double>> Temp::k1;
valarray<complex<double>> Temp::k2;
valarray<complex<double>> Temp::k3;
valarray<complex<double>> Temp::k4;
