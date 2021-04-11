#include <cosmictiger/initial.hpp>
#include <cosmictiger/fourier.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/boltzmann.hpp>
#include <cosmictiger/zero_order.hpp>
#include <cosmictiger/zeldovich.hpp>
#include <cosmictiger/constants.hpp>

#define BOLTZSIZE 256
#define FFTSIZE 256
#define RANDSIZE 256
#define ZELDOSIZE 256

template<class T>
__global__ void vector_free_kernel(vector<T>* vect) {
	if (threadIdx.x == 0) {
		vect->vector < T > ::~vector<T>();
	}
}

void initial_conditions(particle_set& parts) {

	int numBlocks;
	sigma8_integrand *func_ptr;
	zero_order_universe* zeroverse_ptr;
	float* result_ptr;
	cos_state* states;
	cosmic_params params;
	int Nk = 100;
	interp_functor<float>* den_k;
	interp_functor<float>* vel_k;
	cmplx* basis;
	cmplx* phi;
	cmplx* rands;
	const size_t N = global().opts.parts_dim;
	const size_t N3 = sqr(N) * N;

	CUDA_MALLOC(phi, N3);
	CUDA_MALLOC(rands, N3);
	CUDA_MALLOC(basis, N / 2);
	CUDA_MALLOC(den_k, 1);
	CUDA_MALLOC(vel_k, 1);
	CUDA_MALLOC(zeroverse_ptr, 1);
	CUDA_MALLOC(result_ptr, 1);
	CUDA_MALLOC(func_ptr, 1);
	CUDA_MALLOC(states, Nk);

	new (den_k) interp_functor<float>();
	new (vel_k) interp_functor<float>();

	auto& uni = *zeroverse_ptr;

#ifndef __CUDA_ARCH__
	auto den_destroy = den_k->to_device();
	auto vel_destroy = vel_k->to_device();
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
	printf("Computing zero order universe...");
	fflush(stdout);
	create_zero_order_universe(&uni, 1.0e6, params);
	set_zeroverse(&uni);
	printf("Done.\n");
	func_ptr->uni = zeroverse_ptr;
	func_ptr->littleh = params.hubble;
	printf("Computing sigma8 normalization...");
	fflush(stdout);
	float kmin = (1e-4 * params.hubble);
	float kmax = (5 * params.hubble);
	integrate<sigma8_integrand, float> <<<1, BOLTZSIZE>>>(func_ptr,
			(float) std::log(kmin), (float) std::log(kmax), result_ptr, (float) 1.0e-6);
	CUDA_CHECK(cudaDeviceSynchronize());
	*result_ptr = sqrt(sqr(global().opts.sigma8) / *result_ptr);
	printf("Done. Normalization = %e\n", *result_ptr);
	float normalization = *result_ptr;

	int block_size = min(BOLTZSIZE, Nk);
/*		printf("\tComputing Einstain-Boltzmann interpolation solutions for power.dat\n");
	 float dk = log(kmax / kmin) / (Nk - 1);
	 printf("\tComputing Einstain-Boltzmann interpolation solutions for wave numbers %e to %e Mpc^-1\n", kmin, kmax);
	 einstein_boltzmann_interpolation_function<<<1, block_size>>>(den_k, vel_k, states, zeroverse_ptr, kmin, kmax, normalization, Nk, zeroverse_ptr->amin, 1.f);
	 CUDA_CHECK(cudaDeviceSynchronize());
*/
	const auto code_to_mpc = global().opts.code_to_cm / constants::mpc_to_cm;
	printf( "code_to_mpc = %e\n", code_to_mpc);

	kmin = 2.0 * (float) M_PI / code_to_mpc;
	kmax = sqrtf(3) * (kmin * (float) (global().opts.parts_dim));
	printf("\tComputing Einstain-Boltzmann interpolation solutions for wave numbers %e to %e Mpc^-1\n", kmin, kmax);
	Nk = 2 * global().opts.parts_dim;
	block_size = min(BOLTZSIZE, Nk);
	const float ainit = 1.0f / (global().opts.z0 + 1.0f);
	einstein_boltzmann_interpolation_function<<<1, block_size>>>(den_k, vel_k, states, zeroverse_ptr, kmin, kmax, normalization, Nk, zeroverse_ptr->amin, ainit);
	CUDA_CHECK(cudaDeviceSynchronize());

	printf("\tComputing FFT basis\n");
	fft_basis<<<1,FFTSIZE>>>(basis, N);
	CUDA_CHECK(cudaDeviceSynchronize());

	printf("\tComputing random number set\n");
	generate_random_normals<<<1,RANDSIZE>>>(rands, N3);
	CUDA_CHECK(cudaDeviceSynchronize());

	printf("\tComputing over/under density\n");
	zeldovich<<<1,ZELDOSIZE>>>(phi, basis, rands, den_k, code_to_mpc, N, 0, DENSITY);
	CUDA_CHECK(cudaDeviceSynchronize());
	fft3d(phi, basis, N);
	float drho = 0.0;
	for (int i = 0; i < N3; i++) {
		drho = std::max(drho, std::abs((phi[i].real())));
	}
	printf("\t\tOver/under density is %e\n", drho);

	float xdisp = 0.0, vmax = 0.0;
	const double omega_m = params.omega_b + params.omega_c;
	const double omega_r = params.omega_nu + params.omega_gam;
	const double a = ainit;
	const double Om = omega_m / (omega_m + (a * a * a) * (1.0 - omega_m - omega_r) + omega_r / a);
	const double f = std::pow(Om, 0.55);
	const double H = global().opts.H0
			* std::sqrt(omega_r / (a * a * a * a) + omega_m / (a * a * a) + 1.0 - omega_r - omega_m);
	const double prefac = f * H * a;
	printf("Velocity prefactor is %e, Hubble(a) = %e, f(a) = %e\n", prefac, H, f);
	for (int dim = 0; dim < NDIM; dim++) {
		printf("\t\tComputing %c positions\n", 'x' + dim);
		zeldovich<<<1,ZELDOSIZE>>>(phi, basis, rands, den_k, code_to_mpc, N, dim, DISPLACEMENT);
		CUDA_CHECK(cudaDeviceSynchronize());
		fft3d(phi, basis, N);
		for (int i = 0; i < N3; i++) {
			xdisp = std::max(xdisp, std::abs((phi[i].real())));
		}
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				for (int k = 0; k < N; k++) {
					const int l = N * (N * i + j) + k;
					const int I[NDIM] = { i, j, k };
					float x = (((float) I[dim] + 0.5f) / (float) N);
					x += phi[l].real() / code_to_mpc;
					while (x > 1.0) {
						x -= 1.0;
					}
					while (x < 0.0) {
						x += 1.0;
					}
					parts.pos(dim, l) = x;
				}
			}
		}
		printf("\t\tComputing %c velocities\n", 'x' + dim);
		zeldovich<<<1,ZELDOSIZE>>>(phi, basis, rands, den_k, code_to_mpc, N, dim, VELOCITY);
		CUDA_CHECK(cudaDeviceSynchronize());
		fft3d(phi, basis, N);
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				for (int k = 0; k < N; k++) {
					const int l = N * (N * i + j) + k;
					float v = phi[l].real();
					parts.vel(l).a[dim] = v * a / code_to_mpc;
				}
			}
		}
	}
	xdisp /= code_to_mpc / N;
	printf("\t\tMaximum displacement is %e\n", xdisp);

#ifndef __CUDA_ARCH__
	den_destroy();
	vel_destroy();
	cs_destroy();
	sigma_destroy();
#endif
	vector_free_kernel<<<1,1>>>(&vel_k->values);
	vector_free_kernel<<<1,1>>>(&den_k->values);
	vector_free_kernel<<<1,1>>>(&uni.sigma_T.values);
	vector_free_kernel<<<1,1>>>(&uni.cs2.values);
	CUDA_FREE(zeroverse_ptr);
	CUDA_FREE(result_ptr);
	CUDA_FREE(func_ptr);
	CUDA_FREE(states);
	CUDA_FREE(vel_k);
	CUDA_FREE(den_k);
	CUDA_FREE(basis);
	CUDA_FREE(rands);
	CUDA_FREE(phi);
}

