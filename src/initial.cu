#include <cosmictiger/initial.hpp>
#include <cosmictiger/fourier.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/boltzmann.hpp>
#include <cosmictiger/zero_order.hpp>
#include <cosmictiger/zeldovich.hpp>
#include <cosmictiger/constants.hpp>
#include <cosmictiger/particle_sets.hpp>
#include <cosmictiger/power.hpp>

#include <unistd.h>

#define SIGMA8SIZE 256
#define FFTSIZE 256
#define RANDSIZE 256
#define ZELDOSIZE 256

template<class T>
__global__ void vector_free_kernel(vector<T>* vect) {
	if (threadIdx.x == 0) {
		vect->vector < T > ::~vector<T>();
	}
}

__global__ void phi_to_positions(particle_set parts, cmplx* phi, float code_to_mpc, int dim) {
	int i = blockIdx.x;
	int j = blockIdx.y;
	int N = gridDim.x;
	for (int k = threadIdx.x; k < N; k += blockDim.x) {
		const int l = N * (N * i + j) + k;
		const int I[NDIM] = { i, j, k };
		float x = (((float) I[dim] + 0.0f) / (float) N);
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


__global__ void phi_to_shifted_positions(particle_set parts, cmplx* phi, float code_to_mpc, int dim) {
	int i = blockIdx.x;
	int j = blockIdx.y;
	int N = gridDim.x;
	const float inv = 1.f / code_to_mpc;
	for (int k = threadIdx.x; k < N; k += blockDim.x) {
		const int l000 = N * (N * i + j) + k;
		const int l001 = N * (N * i + j) + (k+1)%N;
		const int l010 = N * (N * i + (j+1)%N) + k;
		const int l011 = N * (N * i + (j+1)%N) + (k+1)%N;
		const int l100 = N * (N * ((i+1)%N) + j) + k;
		const int l101 = N * (N * ((i+1)%N) + j) + (k+1)%N;
		const int l110 = N * (N * ((i+1)%N) + (j+1)%N) + k;
		const int l111 = N * (N * ((i+1)%N) + (j+1)%N) + (k+1)%N;
		const int I[NDIM] = { i, j, k };
		float x = (((float) I[dim] + 0.5f) / (float) N);
		x += 0.125f * phi[l000].real() * inv;
		x += 0.125f * phi[l001].real() * inv;
		x += 0.125f * phi[l010].real() * inv;
		x += 0.125f * phi[l011].real() * inv;
		x += 0.125f * phi[l100].real() * inv;
		x += 0.125f * phi[l101].real() * inv;
		x += 0.125f * phi[l110].real() * inv;
		x += 0.125f * phi[l111].real() * inv;
		while (x > 1.0) {
			x -= 1.0;
		}
		while (x < 0.0) {
			x += 1.0;
		}
		parts.pos(dim, l000) = x;
	}
}

__global__ void phi_to_velocities(particle_set parts, cmplx* phi, float a, int dim) {
	int i = blockIdx.x;
	int j = blockIdx.y;
	int N = gridDim.x;
	for (int k = threadIdx.x; k < N; k += blockDim.x) {
		const int l = N * (N * i + j) + k;
		float v = phi[l].real();
		parts.vel(dim, l) = v * a; // / code_to_mpc;
	}
}

__global__ void phi_to_shifted_velocities(particle_set parts, cmplx* phi, float a, int dim) {
	int i = blockIdx.x;
	int j = blockIdx.y;
	int N = gridDim.x;
	for (int k = threadIdx.x; k < N; k += blockDim.x) {
		const int l000 = N * (N * i + j) + k;
		const int l001 = N * (N * i + j) + (k+1)%N;
		const int l010 = N * (N * i + (j+1)%N) + k;
		const int l011 = N * (N * i + (j+1)%N) + (k+1)%N;
		const int l100 = N * (N * ((i+1)%N) + j) + k;
		const int l101 = N * (N * ((i+1)%N) + j) + (k+1)%N;
		const int l110 = N * (N * ((i+1)%N) + (j+1)%N) + k;
		const int l111 = N * (N * ((i+1)%N) + (j+1)%N) + (k+1)%N;
		float v = 0.f;
		v += 0.125f * phi[l000].real();
		v += 0.125f * phi[l001].real();
		v += 0.125f * phi[l010].real();
		v += 0.125f * phi[l011].real();
		v += 0.125f * phi[l100].real();
		v += 0.125f * phi[l101].real();
		v += 0.125f * phi[l110].real();
		v += 0.125f * phi[l111].real();
		parts.vel(dim, l000) = v * a; // / code_to_mpc;
	}
}

#define PHIMAXSIZE 512

__global__ void phi_max_kernel(cmplx* phi, int N3, float* maxes) {
	const int& tid = threadIdx.x;
	const int& bid = blockIdx.x;
	const int& bsz = blockDim.x;
	const int& gsz = gridDim.x;
	__shared__ float local_maxes[PHIMAXSIZE];
	float mymax = 0.0;
	for (int i = tid + bid * bsz; i < N3; i += gsz * bsz) {
		mymax = fmaxf(mymax, phi[i].real());
	}
	local_maxes[tid] = mymax;
	__syncthreads();
	for (int P = bsz / 2; P >= 1; P /= 2) {
		if (tid + P < bsz) {
			local_maxes[tid] = fmaxf(local_maxes[tid], local_maxes[tid + P]);
		}
	}
	__syncthreads();
	if (tid == 0) {
		maxes[bid] = local_maxes[0];
	}
}

float phi_max(cmplx* phi, int N3) {
	int num_blocks;
	CUDA_CHECK(
			cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, generate_random_normals, PHIMAXSIZE, sizeof(float)*PHIMAXSIZE));
	num_blocks *= global().cuda.devices[0].multiProcessorCount;
	num_blocks = std::min(global().cuda.devices[0].maxGridSize[0], num_blocks);
	float* maxes;
	CUDA_MALLOC(maxes, num_blocks);
	phi_max_kernel<<<num_blocks,PHIMAXSIZE>>>(phi,N3,maxes);
	CUDA_CHECK(cudaDeviceSynchronize());
	float num = 0.0;
	for (int i = 0; i < num_blocks; i++) {
		num = std::max(num, maxes[i]);
	}
	CUDA_FREE(maxes);
	return num;
}

void initial_conditions(particle_sets& parts) {

	sigma8_integrand *func_ptr;
	zero_order_universe* zeroverse_ptr;
	float* result_ptr;
	cos_state* states;
	cosmic_params params;
	interp_functor<float>* cdm_k;
	interp_functor<float>* bary_k;
	interp_functor<float>* vel_k;
	cmplx* phi;
	cmplx* rands;
	const size_t N = global().opts.parts_dim;
	int Nk = 8 * 1024;
	const size_t N3 = sqr(N) * N;

	CUDA_MALLOC(phi, N3);
	CUDA_MALLOC(rands, N3);
	CUDA_MALLOC(cdm_k, 1);
	CUDA_MALLOC(bary_k, 1);
	CUDA_MALLOC(vel_k, 1);
	CUDA_MALLOC(zeroverse_ptr, 1);
	CUDA_MALLOC(result_ptr, 1);
	CUDA_MALLOC(func_ptr, 1);
	CUDA_MALLOC(states, Nk);

	new (bary_k) interp_functor<float>();
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
	printf("Computing zero order universe...");
	fflush(stdout);
	create_zero_order_universe(&uni, 1.0e6, params);
	set_zeroverse(&uni);
	printf("Done.\n");
	const auto ns = global().opts.ns;
	func_ptr->uni = zeroverse_ptr;
	func_ptr->littleh = params.hubble;
	func_ptr->ns = ns;
	printf("Computing sigma8 normalization...");
	fflush(stdout);
	float kmin = (1e-4 * params.hubble);
	float kmax = (5 * params.hubble);
	integrate<sigma8_integrand, float> <<<1, SIGMA8SIZE>>>(func_ptr,
			(float) std::log(kmin), (float) std::log(kmax), result_ptr, (float) 1.0e-6);
	CUDA_CHECK(cudaDeviceSynchronize());
	*result_ptr = sqrt(sqr(global().opts.sigma8) / *result_ptr);
	printf("Done. Normalization = %e\n", *result_ptr);
	float normalization = *result_ptr;

	const auto code_to_mpc = global().opts.code_to_cm / constants::mpc_to_cm;
	printf("code_to_mpc = %e\n", code_to_mpc);

	kmin = std::min(kmin, 2.0f * (float) M_PI / (float) code_to_mpc);
	kmax = std::max(2.0f * (float) M_PI / (float) code_to_mpc * (float) (global().opts.parts_dim), kmax);
	printf("\tComputing Einstain-Boltzmann interpolation solutions for wave numbers %e to %e Mpc^-1\n", kmin, kmax);
	const float ainit = 1.0f / (global().opts.z0 + 1.0f);
	einstein_boltzmann_interpolation_function(cdm_k, bary_k, vel_k, states, zeroverse_ptr, kmin, kmax, normalization, Nk,
			zeroverse_ptr->amin, ainit, false, ns);
#ifndef __CUDA_ARCH__
	auto cdm_destroy = cdm_k->to_device();
	auto bary_destroy = bary_k->to_device();
	auto vel_destroy = vel_k->to_device();
#endif

	generate_random_normals<<<32,32>>>(rands, N3,1234);
	CUDA_CHECK(cudaDeviceSynchronize());

	/*	printf("\tComputing over/under density\n");
	 zeldovich<<<1,ZELDOSIZE>>>(phi, rands, den_k, code_to_mpc, N, 0, DENSITY);
	 CUDA_CHECK(cudaDeviceSynchronize());
	 fft3d(phi, N);
	 float drho = phi_max(phi, N3);
	 printf("\t\tMaximum over/under density is %e\n", drho);
	 if (drho > 1.0) {
	 printf("The overdensity is high, consider using an ealier starting redshift\n");
	 } else if (drho < 0.1) {
	 printf("The overdensity is low, consider using a later starting redshift\n");
	 }
	 */
	vector<float> spec(N / 2);
	compute_power_spectrum(phi, spec.data(), N);
	FILE* fp = fopen("power.den", "wt");
	for (int i = 1; i < N / 2; i++) {
		const auto k = 2.0 * M_PI * (i + 0.5) / code_to_mpc;
		fprintf(fp, "%e %e\n", k, spec[i]);
	}
	fclose(fp);

	float xdisp = 0.0;
	const double omega_m = params.omega_b + params.omega_c;
	const double omega_r = params.omega_nu + params.omega_gam;
	const double a = ainit;
	const int num_parts = global().opts.sph ? 2 : 1;
	printf( "%i species\n", num_parts);
	for (int pi = 0; pi < num_parts; pi++) {
		printf( "%s\n", pi == CDM_SET ? "CDM" : "Baryons");
		for (int dim = 0; dim < NDIM; dim++) {
			printf("\t\tComputing %c positions\n", 'x' + dim);
			auto& den_k = pi == CDM_SET ? cdm_k : cdm_k;
			zeldovich<<<1,ZELDOSIZE>>>(phi, rands, den_k, code_to_mpc, N, dim, DISPLACEMENT);
			CUDA_CHECK(cudaDeviceSynchronize());
			fft3d(phi, N);
			const auto this_max = phi_max(phi, N3);
			xdisp = std::max(xdisp, this_max);
			dim3 gdim;
			gdim.x = gdim.y = N;
			gdim.z = 1;
			auto posfunc = /*pi == CDM_SET ?*/ phi_to_positions;// : phi_to_shifted_positions;
			auto velfunc = /*pi == CDM_SET ?*/ phi_to_velocities;// : phi_to_shifted_velocities;
			posfunc<<<gdim,32>>>(parts.sets[pi]->get_virtual_particle_set(), phi,code_to_mpc, dim);
			CUDA_CHECK(cudaDeviceSynchronize());
			printf("\t\tComputing %c velocities\n", 'x' + dim);
			zeldovich<<<1,ZELDOSIZE>>>(phi, rands, vel_k, code_to_mpc, N, dim, VELOCITY);
			CUDA_CHECK(cudaDeviceSynchronize());
			fft3d(phi, N);
			velfunc<<<gdim,32>>>(parts.sets[pi]->get_virtual_particle_set(), phi,a, dim);
			CUDA_CHECK(cudaDeviceSynchronize());
		}
	}
	xdisp /= code_to_mpc / N;
	printf("\t\tMaximum displacement is %e\n", xdisp);

#ifndef __CUDA_ARCH__
	cdm_destroy();
	bary_destroy();
	vel_destroy();
	cs_destroy();
	sigma_destroy();
#endif
	vector_free_kernel<<<1,1>>>(&vel_k->values);
	vector_free_kernel<<<1,1>>>(&cdm_k->values);
	vector_free_kernel<<<1,1>>>(&bary_k->values);
	vector_free_kernel<<<1,1>>>(&uni.sigma_T.values);
	vector_free_kernel<<<1,1>>>(&uni.cs2.values);

	cdm_k->~interp_functor<float>();
	bary_k->~interp_functor<float>();
	vel_k->~interp_functor<float>();
	zeroverse_ptr->~zero_order_universe();
	CUDA_FREE(zeroverse_ptr);
	CUDA_FREE(result_ptr);
	CUDA_FREE(func_ptr);
	CUDA_FREE(states);
	CUDA_FREE(vel_k);
	CUDA_FREE(cdm_k);
	CUDA_FREE(bary_k);
	CUDA_FREE(rands);
	CUDA_FREE(phi);
	free_zeroverse();
	printf("Done initializing\n");
}

