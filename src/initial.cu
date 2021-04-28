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

__device__ static float atomicMax(float* address, float val) {
	int* address_as_i = (int*) address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = ::atomicCAS(address_as_i, assumed, __float_as_int(::fmaxf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}

__device__ float constrain_range(float r) {
	while (r >= 1.0f) {
		r -= 1.f;
	}
	while (r < 0.f) {
		r += 1.f;
	}
	return r;
}

__global__ void phi_to_grid(particle_set parts, cmplx* phi, float code_to_mpc, float D1, float prefactor,
		float* maxdisp) {
	int i = blockIdx.x;
	int j = blockIdx.y;
	int N = gridDim.x;
	const float Ninv = 1.0 / (float) N;
	const float unitinv = 1.0 / code_to_mpc;
	__shared__ float maxes[32];
	const float dxinv = N * unitinv;
	const auto tid = threadIdx.x;
	maxes[tid] = 0.f;
	__syncthreads();
	for (int k = tid; k < N; k += blockDim.x) {
		const int l = N * (N * i + j) + k;
		const int lpx = N * (N * ((i + 1) % N) + j) + k;
		const int lpy = N * (N * i + ((j + 1) % N)) + k;
		const int lpz = N * (N * i + j) + (k + 1) % N;
		const int lmx = N * (N * ((i + N - 1) % N) + j) + k;
		const int lmy = N * (N * i + ((j + N - 1) % N)) + k;
		const int lmz = N * (N * i + j) + (k + N - 1) % N;
		array<float, NDIM> x, dphidx;
		dphidx[0] = 0.5f * (phi[lpx].real() - phi[lmx].real()) * dxinv;
		dphidx[1] = 0.5f * (phi[lpy].real() - phi[lmy].real()) * dxinv;
		dphidx[2] = 0.5f * (phi[lpz].real() - phi[lmz].real()) * dxinv;
		x[0] = (i + 0.5f) * Ninv;
		x[1] = (j + 0.5f) * Ninv;
		x[2] = (k + 0.5f) * Ninv;
		for (int dim = 0; dim < NDIM; dim++) {
			float this_disp = D1 * dphidx[dim] * unitinv;
			maxes[tid] = fmaxf(maxes[tid], fabs(this_disp * N));
			const float x0 = constrain_range(x[dim] - this_disp);
			parts.pos(dim, l) = x0;
			parts.vel(dim, l) = -prefactor * this_disp;
		}
	}
	for (int P = blockDim.x / 2; P >= 1; P /= 2) {
		__syncthreads();
		if (tid + P < blockDim.x) {
			maxes[tid] = fmaxf(maxes[tid], maxes[tid + P]);
		}
	}
	atomicMax(maxdisp, maxes[0]);
}

float growth_factor(float omega_m, float a) {
	const float omega_l = 1.f - omega_m;
	const float a3 = a * sqr(a);
	const float deninv = 1.f / (omega_m + a3 * omega_l);
	const float Om = omega_m * deninv;
	const float Ol = a3 * omega_l * deninv;
	return a * 2.5 * Om / (pow(Om, 4.f / 7.f) - Ol + (1.f + 0.5f * Om) * (1.f + 0.014285714f * Ol));
}

__global__
void create_overdensity_transform(cmplx* phi, const cmplx* rands, const interp_functor<float>* Pptr, float box_size,
		int N) {
	const int& thread = threadIdx.x;
	const int& block_size = blockDim.x;
	auto& P = *Pptr;
	for (int i = thread; i < N * N * N; i += block_size) {
		phi[i] = cmplx(0, 0);
	}
	__syncthreads();
	for (int ij = thread; ij < N * N; ij += block_size) {
		int i = ij / N;
		int j = ij % N;
		int i0 = i < N / 2 ? i : i - N;
		int j0 = j < N / 2 ? j : j - N;
		float kx = 2.f * (float) M_PI / box_size * float(i0);
		float ky = 2.f * (float) M_PI / box_size * float(j0);
		for (int l = 0; l < N / 2; l++) {
			int l0 = l < N / 2 ? l : l - N;
			int i2 = i0 * i0 + j0 * j0 + l0 * l0;
			int index0 = N * (N * i + j) + l;
			int index1;
			index1 = N * (N * ((N - i) % N) + ((N - j) % N)) + ((N - l) % N);
			if (i2 > 0 && i2 < N * N / 4) {
				float kz = 2.f * (float) M_PI / box_size * float(l0);
				float k = sqrt(kx * kx + ky * ky + kz * kz);
				phi[index0] = rands[index0] * sqrtf(P(k)) * powf(box_size, -1.5);
				phi[index1] = phi[index0].conj();
			}
		}
	}
	__syncthreads();
}

__global__
void transform_laplacian(cmplx* phi, float box_size, int N) {
	const int& thread = threadIdx.x;
	const int& block_size = blockDim.x;
	__syncthreads();
	for (int ij = thread; ij < N * N; ij += block_size) {
		int i = ij / N;
		int j = ij % N;
		int i0 = i < N / 2 ? i : i - N;
		int j0 = j < N / 2 ? j : j - N;
		float kx = 2.f * (float) M_PI / box_size * float(i0);
		float ky = 2.f * (float) M_PI / box_size * float(j0);
		for (int l = 0; l < N; l++) {
			int l0 = l < N / 2 ? l : l - N;
			int i2 = i0 * i0 + j0 * j0 + l0 * l0;
			int index0 = N * (N * i + j) + l;
			if (i2 > 0 && i2 < N * N / 4) {
				float kz = 2.f * (float) M_PI / box_size * float(l0);
				float k2 = (kx * kx + ky * ky + kz * kz);
				phi[index0].real() /= -k2;
				phi[index0].imag() /= -k2;
			}
		}
	}
	__syncthreads();
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
	cmplx* phi1;
	cmplx* rands;
	const size_t N = global().opts.parts_dim;
	int Nk = 1024;
	const size_t N3 = sqr(N) * N;
	float* max_disp;

	CUDA_MALLOC(max_disp, 1);
	CUDA_MALLOC(phi1, N3);
	CUDA_MALLOC(rands, N3);
	CUDA_MALLOC(cdm_k, 1);
	CUDA_MALLOC(bary_k, 1);
	CUDA_MALLOC(vel_k, 1);
	CUDA_MALLOC(zeroverse_ptr, 1);
	CUDA_MALLOC(result_ptr, 1);
	CUDA_MALLOC(func_ptr, 1);
	CUDA_MALLOC(states, Nk);

	*max_disp = 0.f;
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
			zeroverse_ptr->amin, 1.0, false, ns);
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
	compute_power_spectrum(phi1, spec.data(), N);
	FILE* fp = fopen("power.den", "wt");
	for (int i = 1; i < N / 2; i++) {
		const auto k = 2.0 * M_PI * (i + 0.5) / code_to_mpc;
		fprintf(fp, "%e %e\n", k, spec[i]);
	}
	fclose(fp);

	float xdisp = 0.0;
	const double omega_m = params.omega_b + params.omega_c;
	const double a = ainit;
	const int num_parts = global().opts.sph ? 2 : 1;
	printf("%i species\n", num_parts);
	const double Om = omega_m / (omega_m + (a * a * a) * (1.0 - omega_m));
	const float D1 = growth_factor(omega_m, a) / growth_factor(omega_m, 1.0);
	const double f1 = std::pow(Om, 5.f / 9.f);
	const double H = global().opts.H0 * global().opts.hubble * std::sqrt(omega_m / (a * a * a) + 1.0 - omega_m);
	double prefac = f1 * H * a;
	printf("D1 = %e\n", D1);
	printf("H = %e\n", H);
	printf("f1 = %e\n", f1);
	printf("H*a*f1 = %e\n", prefac);
	printf("\t\tComputing positions\n");
	auto& den_k = cdm_k;
	create_overdensity_transform<<<1,ZELDOSIZE>>>(phi1, rands, den_k, code_to_mpc, N);
	CUDA_CHECK(cudaDeviceSynchronize());
	transform_laplacian<<<1,ZELDOSIZE>>>(phi1, code_to_mpc, N);
	CUDA_CHECK(cudaDeviceSynchronize());
	fft3d(phi1, N);
	dim3 gdim;
	gdim.x = gdim.y = N;
	gdim.z = 1;
	phi_to_grid<<<gdim,32>>>(parts.cdm.get_virtual_particle_set(),phi1, code_to_mpc,D1, a*prefac,max_disp);
	CUDA_CHECK(cudaDeviceSynchronize());
	printf("\t\tMaximum displacement is %e\n", *max_disp);

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
	CUDA_FREE(phi1);
	CUDA_FREE(max_disp);
	free_zeroverse();
	printf("Done initializing\n");
}

