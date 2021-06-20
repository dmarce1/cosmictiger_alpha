#include <cosmictiger/initial.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/zeldovich.hpp>
#include <cosmictiger/constants.hpp>
#include <cosmictiger/particle_server.hpp>
#include <cosmictiger/power.hpp>

#include <unistd.h>

#define SIGMA8SIZE 256
#define FFTSIZE 256
#define RANDSIZE 256
#define ZELDOSIZE 1024

struct sigma8_integrand {
	interp_functor<float> power;
	float littleh;
	__device__ float operator()(float x) const;
};



double growth_factor(double omega_m, float a) {
	const double omega_l = 1.f - omega_m;
	const double a3 = a * sqr(a);
	const double deninv = 1.f / (omega_m + a3 * omega_l);
	const double Om = omega_m * deninv;
	const double Ol = a3 * omega_l * deninv;
	return a * 2.5 * Om / (pow(Om, 4.f / 7.f) - Ol + (1.f + 0.5f * Om) * (1.f + 0.014285714f * Ol));
}

interp_functor<float> read_power_spectrum() {
	interp_functor<float> functor;
	const float h = global().opts.hubble;
	;
	const float h3 = h * h * h;
	FILE* fp = fopen("power.init", "rb");
	if (fp == nullptr) {
		printf("Cannot read power.init, unable to continue.\n");
		fflush(stdout);
		abort();
	}
	float kmax = 0.0;
	float kmin = std::numeric_limits<float>::max();
	vector<float> P;
	while (!feof(fp)) {
		float k, p;
		if (fscanf(fp, "%f %f\n", &k, &p)) {
			k *= h;
			p /= h3;
			kmax = std::max(kmax, k);
			kmin = std::min(kmin, k);
			P.push_back(p);
		}
	}
//	printf( "%e %e\n", kmin, kmax);
	fclose(fp);
	build_interpolation_function(&functor, P, kmin, kmax);
	return std::move(functor);
}

__device__ float sigma8_integrand::operator()(float x) const {
	const float R = 8 / littleh;
	const float c0 = float(9) / (2. * float(M_PI) * float(M_PI)) / powf(R, 6);
	float k = expf(x);
	float P = power(k);
	float tmp = (SIN(k*R) - k * R * COS(k * R));
	return c0 * P * tmp * tmp * powf(k, -3);
}

void initial_conditions(particle_set& parts) {
	{
		const size_t N = global().opts.parts_dim;

		interp_functor<float> power = read_power_spectrum();
		float kmin = power.amin;
		float kmax = power.amax;

		const auto code_to_mpc = global().opts.code_to_cm / constants::mpc_to_cm;
		float kmin_max = 2.0f * (float) M_PI / (float) code_to_mpc;
		float kmax_min = (float) M_PI / (float) code_to_mpc * (float) N;
		if (kmin > kmin_max) {
			PRINT("kmin = %e kmin_max = %e\n", kmin, kmin_max);
			PRINT("%s", "kmin of power.init is too large, unable to continue.\n");
			abort();
		}
		if (kmax < kmax_min) {
			PRINT("kmax = %e kmax_min = %e\n", kmax, kmax_min);
			PRINT("%s", "kmax of power.init is too small, unable to continue\n");
			abort();
		}
		PRINT("code_to_mpc = %e\n", code_to_mpc);

		sigma8_integrand* sigma8_func;
		float* result_ptr;
		CUDA_MALLOC(sigma8_func, 1);
		CUDA_MALLOC(result_ptr, 1);
		sigma8_func->littleh = global().opts.hubble;
		sigma8_func->power = power;

		PRINT("%s", "Computing sigma8 normalization...");
		integrate<sigma8_integrand, float> <<<1, SIGMA8SIZE>>>(sigma8_func,
				(float) std::log(kmin*1.001), (float) std::log(kmax/1.001), result_ptr, (float) 1.0e-6);
		CUDA_CHECK(cudaDeviceSynchronize());
		*result_ptr = sqrt(sqr(global().opts.sigma8) / *result_ptr);
		PRINT("Done. Normalization = %e\n", *result_ptr);
		float normalization = *result_ptr;

		{
			vector<float> new_power(power.values.size());
			for (int i = 0; i < power.values.size(); i++) {
				new_power[i] = normalization * power.values[i];
			}
			build_interpolation_function(&power, new_power, power.amin, power.amax);
		}

		const double omega_m = global().opts.omega_b + global().opts.omega_c;
		const double a = 1.0 / (global().opts.z0 + 1.0);
		const double Om = omega_m / (omega_m + (a * a * a) * (1.0 - omega_m));
		const double D1 = growth_factor(omega_m, a) / growth_factor(omega_m, 1.0);
		const double D2 = -3.f * sqr(D1) / 7.f;
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
		PRINT("%s", "\t\tComputing positions\n");

		int seed = 1234;
		float max_disp = 0.0;
		for (int dim = 0; dim < NDIM; dim++) {
			printf("Computing order 1 %c positions and %c velocities\n", 'x' + dim, 'x' + dim);
			_2lpt(power, N, code_to_mpc, dim, NDIM, seed);
			max_disp = std::max(max_disp, phi1_to_particles(N, code_to_mpc, D1, a * prefac1, dim));
		}
		bool use_2lpt = false;
		printf("Maximum displacement is %e\n", max_disp);
		if (use_2lpt) {
			printf("2LPT phase 1\n");
			_2lpt_init(N);
			_2lpt(power, N, code_to_mpc, 0, 0, seed);
			_2lpt_phase(N, 0);
			printf("2LPT phase 2\n");
			_2lpt(power, N, code_to_mpc, 1, 1, seed);
			_2lpt_phase(N, 1);
			printf("2LPT phase 3\n");
			_2lpt(power, N, code_to_mpc, 0, 0, seed);
			_2lpt_phase(N, 2);
			printf("2LPT phase 4\n");
			_2lpt(power, N, code_to_mpc, 2, 2, seed);
			_2lpt_phase(N, 3);
			printf("2LPT phase 5\n");
			_2lpt(power, N, code_to_mpc, 1, 1, seed);
			_2lpt_phase(N, 4);
			printf("2LPT phase 6\n");
			_2lpt(power, N, code_to_mpc, 2, 2, seed);
			_2lpt_phase(N, 5);
			printf("2LPT phase 7\n");
			_2lpt(power, N, code_to_mpc, 0, 1, seed);
			_2lpt_phase(N, 6);
			printf("2LPT phase 8\n");
			_2lpt(power, N, code_to_mpc, 0, 2, seed);
			_2lpt_phase(N, 7);
			printf("2LPT phase 9\n");
			_2lpt(power, N, code_to_mpc, 1, 2, seed);
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
		hpxfft::fourier3d_destroy();
		sigma8_func->power.values.~vector<float>();
		CUDA_FREE(sigma8_func);
		CUDA_FREE(result_ptr);

	}

	unified_allocator alloc;
	alloc.reset();
	printf( "Doing initial domain decomposition\n");
	particle_server pserv;
	pserv.apply_domain_decomp();
	printf("Computing initial matter power spectrum\n");
	matter_power_spectrum(0);
	alloc.reset();


	PRINT("%s", "Done initializing\n");
}

