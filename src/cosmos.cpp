#include <cosmictiger/cosmos.hpp>
#include <cosmictiger/global.hpp>

#include <cmath>

double cosmos_dadtau(double a) {
	const auto H = global().opts.H0 * global().opts.hubble;
	const auto omega_m = global().opts.omega_m;
	const auto omega_r = global().opts.omega_gam + global().opts.omega_nu;
	const auto omega_lambda = 1.0 - omega_m - omega_r;
#ifdef CONFORMAL_TIME
	return H * a * a * std::sqrt(omega_r / (a * a * a * a) + omega_m / (a * a * a) + omega_lambda);
#else
	return H * a * std::sqrt(omega_r / (a * a * a * a) + omega_m / (a * a * a) + omega_lambda);
#endif
}

double cosmos_drift_dtau(double a, double t0) {
	int N = 100;
	double dt = t0 / N;
	double drift_dt = 0.0;
	double t = 0.0;
	while (t < t0) {
		const double dadt1 = cosmos_dadtau(a);
		const double dadt2 = cosmos_dadtau(a + dadt1 * dt);
#ifdef CONFORMAL_TIME
		drift_dt += 0.5 / a / N;
#else
		drift_dt += 0.5 / (a*a) / N;
#endif
		a += 0.5 * (dadt1 + dadt2) * dt;
#ifdef CONFORMAL_TIME
		drift_dt += 0.5 / a / N;
#else
		drift_dt += 0.5 / (a*a) / N;
#endif
		t += dt;
	}
	return drift_dt;
}

double cosmos_age(double a0) {
	double a = a0;
	double t = 0.0;
	while (a < 1.0) {
		const double dadt1 = cosmos_dadtau(a);
		const double dt = (a / dadt1) * 1.e-5;
		const double dadt2 = cosmos_dadtau(a + dadt1 * dt);
		a += 0.5 * (dadt1 + dadt2) * dt;
		t += dt;
	}
	return t;
}
