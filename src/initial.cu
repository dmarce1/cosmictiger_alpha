#include <cosmictiger/initial.hpp>
#include <cosmictiger/fourier.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/boltzmann.hpp>
#include <cosmictiger/zero_order.hpp>
#include <cosmictiger/zeldovich.hpp>
#include <cosmictiger/constants.hpp>
#include <cosmictiger/particle.hpp>
#include <cosmictiger/power.hpp>

#include <unistd.h>

#define SIGMA8SIZE 256
#define FFTSIZE 256
#define RANDSIZE 256
#define ZELDOSIZE 1024

template<class T>
__global__ void vector_free_kernel(vector<T>* vect) {
	if (threadIdx.x == 0) {
		vect->vector < T > ::~vector<T>();
	}
}

float growth_factor(float omega_m, float a) {
	const float omega_l = 1.f - omega_m;
	const float a3 = a * sqr(a);
	const float deninv = 1.f / (omega_m + a3 * omega_l);
	const float Om = omega_m * deninv;
	const float Ol = a3 * omega_l * deninv;
	return a * 2.5 * Om / (pow(Om, 4.f / 7.f) - Ol + (1.f + 0.5f * Om) * (1.f + 0.014285714f * Ol));
}

void initial_conditions(particle_set& parts) {

	sigma8_integrand *func_ptr;
	zero_order_universe* zeroverse_ptr;
	float* result_ptr;
	cos_state* states;
	cosmic_params params;
	interp_functor<float>* cdm_k;
	interp_functor<float>* vel_k;
	cmplx* phi1;
	cmplx* phi2;
	const size_t N = global().opts.parts_dim;
	int Nk = 1024;
	const size_t N3 = sqr(N) * N;
	float max_disp;

	CUDA_MALLOC(phi1, N3);
	CUDA_MALLOC(phi2, N3);
	CUDA_MALLOC(cdm_k, 1);
	CUDA_MALLOC(vel_k, 1);
	CUDA_MALLOC(zeroverse_ptr, 1);
	CUDA_MALLOC(result_ptr, 1);
	CUDA_MALLOC(func_ptr, 1);
	CUDA_MALLOC(states, Nk);

	new (cdm_k) interp_functor<float>();
	new (vel_k) interp_functor<float>();

	auto& uni = *zeroverse_ptr;

#ifndef __CUDA_ARCH__
	auto cs_destroy = uni.cs2.to_device();
	auto sigma_destroy = uni.sigma_T.to_device();
#endif
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
	set_zeroverse(&uni);
	PRINT("Done.\n");
	const auto ns = global().opts.ns;
	func_ptr->uni = zeroverse_ptr;
	func_ptr->littleh = params.hubble;
	func_ptr->ns = ns;
	PRINT("Computing sigma8 normalization...");
	fflush(stdout);
	float kmin = (1e-4 * params.hubble);
	float kmax = (5 * params.hubble);
	integrate<sigma8_integrand, float> <<<1, SIGMA8SIZE>>>(func_ptr,
			(float) std::log(kmin), (float) std::log(kmax), result_ptr, (float) 1.0e-6);
	CUDA_CHECK(cudaDeviceSynchronize());
	*result_ptr = sqrt(sqr(global().opts.sigma8) / *result_ptr);
	PRINT("Done. Normalization = %e\n", *result_ptr);
	float normalization = *result_ptr;

	const auto code_to_mpc = global().opts.code_to_cm / constants::mpc_to_cm;
	PRINT("code_to_mpc = %e\n", code_to_mpc);

	kmin = std::min(kmin, 2.0f * (float) M_PI / (float) code_to_mpc);
	kmax = std::max((float) M_PI / (float) code_to_mpc * (float) (global().opts.parts_dim), kmax);
	PRINT("\tComputing Einstain-Boltzmann interpolation solutions for wave numbers %e to %e Mpc^-1\n", kmin, kmax);
	const float ainit = 1.0f / (global().opts.z0 + 1.0f);
	einstein_boltzmann_interpolation_function(cdm_k, vel_k, states, zeroverse_ptr, kmin, kmax, normalization, Nk,
			zeroverse_ptr->amin, 1.0, false, ns);
#ifndef __CUDA_ARCH__
	auto cdm_destroy = cdm_k->to_device();
	auto vel_destroy = vel_k->to_device();
#endif

	/*	vector<float> spec(N / 2);
	 compute_power_spectrum(phi1, spec.data(), N);
	 FILE* fp = fopen("power.den", "wt");
	 for (int i = 1; i < N / 2; i++) {
	 const auto k = 2.0 * M_PI * (i + 0.5) / code_to_mpc;
	 fprintf(fp, "%e %e\n", k, spec[i]);
	 }
	 fclose(fp);*/

	const double omega_m = params.omega_b + params.omega_c;
	const double a = ainit;
	const double Om = omega_m / (omega_m + (a * a * a) * (1.0 - omega_m));
	const float D1 = growth_factor(omega_m, a) / growth_factor(omega_m, 1.0);
	const float D2 = -3.f * sqr(D1) / 7.f;
	const double f1 = std::pow(Om, 5.f / 9.f);
	const double f2 = 2.f * std::pow(Om, 6.f / 11.f);
	const double H = global().opts.H0 * global().opts.hubble * std::sqrt(omega_m / (a * a * a) + 1.0 - omega_m);
	double prefac1 = f1 * H * a;
	double prefac2 = f2 * H * a;
	PRINT("D1 = %e\n", D1);
	PRINT("D2 = %e\n", D2);
	PRINT("H = %e\n", H);
	PRINT("f1 = %e\n", f1);
	PRINT("f2 = %e\n", f2);
	PRINT("H*a*f1 = %e\n", prefac1);
	PRINT("H*a*f2 = %e\n", prefac2);
	PRINT("\t\tComputing positions\n");

	auto& den_k = cdm_k;
	int seed = time(NULL)+42;
	max_disp = 0.0;
	for (int dim = 0; dim < NDIM; dim++) {
		printf("Computing order 1 %c positions and %c velocities\n", 'x' + dim, 'x' + dim);
		_2lpt(*den_k, N, code_to_mpc, dim, NDIM, seed);
		max_disp = std::max(max_disp, phi1_to_particles(N, code_to_mpc, D1, a * prefac1, dim));
	}
	bool use_2lpt = true;
	printf("Maximum displacement is %e\n", max_disp);
	if (use_2lpt) {
		printf("2LPT phase 1\n");
		_2lpt_init(N);
		_2lpt(*den_k, N, code_to_mpc, 0, 0, seed);
		_2lpt_phase(N, 0);
		printf("2LPT phase 2\n");
		_2lpt(*den_k, N, code_to_mpc, 1, 1, seed);
		_2lpt_phase(N, 1);
		printf("2LPT phase 3\n");
		_2lpt(*den_k, N, code_to_mpc, 0, 0, seed);
		_2lpt_phase(N, 2);
		printf("2LPT phase 4\n");
		_2lpt(*den_k, N, code_to_mpc, 2, 2, seed);
		_2lpt_phase(N, 3);
		printf("2LPT phase 5\n");
		_2lpt(*den_k, N, code_to_mpc, 1, 1, seed);
		_2lpt_phase(N, 4);
		printf("2LPT phase 6\n");
		_2lpt(*den_k, N, code_to_mpc, 2, 2, seed);
		_2lpt_phase(N, 5);
		printf("2LPT phase 7\n");
		_2lpt(*den_k, N, code_to_mpc, 0, 1, seed);
		_2lpt_phase(N, 6);
		printf("2LPT phase 8\n");
		_2lpt(*den_k, N, code_to_mpc, 0, 2, seed);
		_2lpt_phase(N, 7);
		printf("2LPT phase 9\n");
		_2lpt(*den_k, N, code_to_mpc, 1, 2, seed);
		_2lpt_phase(N, 8);
		printf("Computing 2LPT correction\n");
		_2lpt_correction1(N, code_to_mpc);
		max_disp = 0.0;
		for (int dim = 0; dim < NDIM; dim++) {
			printf("Computing 2LPT correction to %c positions and velocities\n", 'x' + dim);
			_2lpt_correction2(N, code_to_mpc, dim);
			max_disp = std::max(max_disp, phi2_to_particles(N, code_to_mpc, D2, a * prefac2, dim));
		}
		printf("Maxmimum correction = %e\n", max_disp);
		_2lpt_destroy();
	}
	fourier3d_destroy();
	printf( "Computing initial matter power spectrum\n");
	matter_power_spectrum(0);


#ifndef __CUDA_ARCH__
	cdm_destroy();
	vel_destroy();
	cs_destroy();
	sigma_destroy();
#endif
	vector_free_kernel<<<1,1>>>(&vel_k->values);
	vector_free_kernel<<<1,1>>>(&cdm_k->values);
	vector_free_kernel<<<1,1>>>(&uni.sigma_T.values);
	vector_free_kernel<<<1,1>>>(&uni.cs2.values);

	cdm_k->~interp_functor<float>();
	vel_k->~interp_functor<float>();
	zeroverse_ptr->~zero_order_universe();
	CUDA_FREE(zeroverse_ptr);
	CUDA_FREE(result_ptr);
	CUDA_FREE(func_ptr);
	CUDA_FREE(states);
	CUDA_FREE(vel_k);
	CUDA_FREE(cdm_k);
	CUDA_FREE(phi1);
	CUDA_FREE(phi2);
	free_zeroverse();
	unified_allocator alloc;
	alloc.reset();
	PRINT("Allocator reset\n");

	PRINT("Done initializing\n");
}

