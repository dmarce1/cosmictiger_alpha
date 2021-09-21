#include <cosmictiger/initial.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/boltzmann.hpp>
#include <cosmictiger/zero_order.hpp>
#include <cosmictiger/constants.hpp>

#include <unistd.h>

#define SIGMA8SIZE 256
#define FFTSIZE 256
#define RANDSIZE 256
#define ZELDOSIZE 1024

float growth_factor(float omega_m, float a) {
	const float omega_l = 1.f - omega_m;
	const float a3 = a * sqr(a);
	const float deninv = 1.f / (omega_m + a3 * omega_l);
	const float Om = omega_m * deninv;
	const float Ol = a3 * omega_l * deninv;
	return a * 2.5 * Om / (pow(Om, 4.f / 7.f) - Ol + (1.f + 0.5f * Om) * (1.f + 0.014285714f * Ol));
}

void initial_conditions() {

	zero_order_universe zeroverse;
	float* result_ptr;
	cosmic_params params;
	interp_functor<float> cdm_k;
	interp_functor<float> vel_k;
	const size_t N = global().opts.parts_dim;
	int Nk = 1024;
	cos_state states[Nk];
	const size_t N3 = sqr(N) * N;

	auto& uni = zeroverse;
	params.omega_b = global().opts.omega_b;
	params.omega_c = global().opts.omega_c;
	params.omega_gam = global().opts.omega_gam;
	params.omega_nu = global().opts.omega_nu;
	params.Y = global().opts.Y;
	params.Neff = global().opts.Neff;
	params.Theta = global().opts.Theta;
	params.hubble = global().opts.hubble;
	PRINT("Computing zero order universe...");
	fflush(stdout);
	create_zero_order_universe(&uni, 1.0e6, params);
	PRINT("Done.\n");
	const auto ns = global().opts.ns;
	fflush(stdout);
	float kmin;
	float kmax;
	kmin = 1e-4 * params.hubble;
	kmax = 25.271 * params.hubble;
	PRINT("\tComputing Einstain-Boltzmann interpolation solutions for wave numbers %e to %e Mpc^-1\n", kmin, kmax);
	einstein_boltzmann_interpolation_function(&cdm_k, &vel_k, states, &zeroverse, kmin, kmax, 1.0, Nk,
			zeroverse.amin, 1.0, false, ns);

}

