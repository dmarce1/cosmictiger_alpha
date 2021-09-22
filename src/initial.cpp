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

double growth_factor(double omega_m, double a) {
	const double omega_l = 1.f - omega_m;
	const double a3 = a * sqr(a);
	const double deninv = 1.f / (omega_m + a3 * omega_l);
	const double Om = omega_m * deninv;
	const double Ol = a3 * omega_l * deninv;
	return a * 2.5 * Om / (pow(Om, 4.f / 7.f) - Ol + (1.f + 0.5f * Om) * (1.f + 0.014285714f * Ol));
}

void initial_conditions() {

	zero_order_universe zeroverse;
	double* result_ptr;
	cosmic_params params;
	interp_functor<double> m_k;
	interp_functor<double> vel_k;
	int Nk = 1024;
	cos_state states[Nk];

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
	double kmin;
	double kmax;
	kmin = 1e-4 * params.hubble;
	kmax = 25.271 * params.hubble;
	einstein_boltzmann_interpolation_function(&m_k, &vel_k, states, &zeroverse, kmin, kmax, 1.0, Nk, zeroverse.amin, 1.0,
			false, ns);

	const auto sigma8_integrand = [params,m_k](double x) {
		double R = 8 / params.hubble;
		const double c0 = double(9) / (2. * double(M_PI) * double(M_PI)) / powf(R, 6);
		double k = std::exp(x);
		double P = m_k(k);
		double tmp = (std::sin(k * R) - k * R * std::cos(k * R));
		return c0 * P * tmp * tmp * std::pow(k, -3);
	};

	const int M = 2 * Nk;
	const double logkmin = log(kmin);
	const double logkmax = log(kmax);
	const double dlogk = (logkmax - logkmin) / M;
	double sum = 0.0;
	for (int i = 0; i < M; i++) {
		const double logka = logkmin + i * dlogk;
		const double logkb = logkmin + (i + 0.5) * dlogk;
		const double logkc = logkmin + (i + 1) * dlogk;
		const double Ia = sigma8_integrand(logka);
		const double Ib = sigma8_integrand(logkb);
		const double Ic = sigma8_integrand(logkc);
		sum += ((1.0 / 6.0) * (Ia + Ic) + (2.0 / 3.0) * Ib) * dlogk;
	}
	const double norm = sqr(global().opts.sigma8) / sum;
	PRINT( "Normalization = %e\n", norm);
	FILE* fp = fopen("power.dat", "wt");
	const double lh = params.hubble;
	const double lh3 = lh * lh * lh;
	for (int i = 0; i < M; i++) {
		double k = exp(logkmin + (double) i * dlogk);
		fprintf(fp, "%e %e %e\n", k / lh, norm * m_k(k) * lh3, norm * vel_k(k) * lh3);
	}
	fclose(fp);

}

