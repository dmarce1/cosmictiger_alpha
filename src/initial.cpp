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

#define RECFAST_N 1000
#define RECFAST_Z0 9990
#define RECFAST_Z1 0
#define RECFAST_DZ 10


std::function<double(double)> run_recfast(cosmic_params params) {
	std::function<double(double)> func;
	FILE* fp = fopen("recfast.out", "rb");
	if (fp != NULL) {
		fclose(fp);
		if (system("rm recfast.out") != 0) {
			printf("Unable to erase recfast.out\n");
			abort();
		}
	}
	fp = fopen("recfast.in", "wb");
	if (fp == NULL) {
		printf("Unable to write to recfast.in\n");
		abort();
	}
	fprintf(fp, "recfast.out\n");
	fprintf(fp, "%f %f %f\n", params.omega_b, params.omega_c, 1.0 - params.omega_b - params.omega_c);
	fprintf(fp, "%f %f %f\n", 100 * params.hubble, params.Theta * 2.73, params.Y);
	fprintf(fp, "1\n");
	fprintf(fp, "6\n");
	fclose(fp);
	if (system("cat recfast.in | ./recfast 1> /dev/null 2> /dev/null") != 0) {
		printf("Unable to run RECFAST\n");
		abort();
	}
	fp = fopen("recfast.out", "rb");
	char d1[2];
	char d2[4];
	if (fscanf(fp, " %s %s\n", d1, d2) == 0) {
		printf("unable to read recfast.out\n");
		abort();
	}
	std::vector<double> xe;
	for (int i = 0; i < RECFAST_N; i++) {
		float z;
		float this_xe;
		if (fscanf(fp, "%f %f\n", &z, &this_xe) == 0) {
			printf("unable to read recfast.out\n");
			abort();
		}
		xe.push_back(this_xe);
	}
	std::vector<double> tmp;
	for (int i = 0; i < RECFAST_N; i++) {
		tmp.push_back(xe.back());
		xe.pop_back();
	}
	xe = std::move(tmp);
	fclose(fp);
	auto inter_func = [xe](double a) {
		const double z = 1.0 / a - 1.0;
		const int i1 = std::min(std::max((int)(z / RECFAST_DZ),1),RECFAST_N-2);
		if( i1 == RECFAST_N - 2 ) {
			return (double) xe.back();
		} else {
			const int i0 = i1 - 1;
			const int i2 = i1 + 1;
			const int i3 = i1 + 2;
			const double t = z / RECFAST_DZ - i1;
			const double y0 = xe[i0];
			const double y1 = xe[i1];
			const double y2 = xe[i2];
			const double y3 = xe[i3];
			const double ct = t * (1.0 - t);
			const double d = -0.5 * ct * ((1.0 - t) * y0 + t * y3);
			const double b = (1.0 - t + ct * (1.0 - 1.5 * t)) * y1;
			const double c = (t + ct * (1.5 * t - 0.5)) * y2;
			return d + b + c;
		}
	};
	return std::move(inter_func);
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
	auto fxe = run_recfast(params);
	create_zero_order_universe(&uni, fxe, 1.1, params);
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

