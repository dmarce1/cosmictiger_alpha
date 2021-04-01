#include <cosmictiger/zeroverse.hpp>
#include <cosmictiger/vector.hpp>
#include <cosmictiger/constants.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/math.hpp>
#include <functional>

#include <cmath>

#define YHelium 0.74


template<class T>

void build_interpolation_function(interp_functor<T>* f, const vector<T>& values, T amin, T amax) {
	T minloga = log(amin);
	T maxloga = log(amax);
	int N = values.size() - 1;
	T dloga = (maxloga - minloga) / N;
	interp_functor<T> functor;
	functor.values = std::move(values);
	functor.maxloga = maxloga;
	functor.minloga = minloga;
	functor.dloga = dloga;
	functor.amin = amin;
	functor.amax = amax;
	functor.N = N;
	*f = functor;
}

template<class T>

void build_interpolation_function(interp_functor<T>* f, T* values, T amin, T amax, int N) {
	T minloga = log(amin);
	T maxloga = log(amax);
	T dloga = (maxloga - minloga) / N;
	interp_functor<T> functor;
	functor.values.resize(N);
	for (int i = 0; i < N; i++) {
		functor.values[i] = values[i];
	}
	functor.maxloga = maxloga;
	functor.minloga = minloga;
	functor.dloga = dloga;
	functor.amin = amin;
	functor.amax = amax;
	functor.N = N;
	*f = functor;
}

void print_time(double tm) {
	const double speryear = 365.24 * 24 * 3600.0;
	if (tm < 60.0) {
		printf("%.1f seconds", tm);
	} else if (tm < 3600.0) {
		printf("%.1f minutes", tm / 60);
	} else if (tm < 24 * 3600.0) {
		printf("%.1f hours", tm / 3600);
	} else if (tm < 365.24 * 24 * 3600.0) {
		printf("%.1f days", tm / 3600 / 24);
	} else if (tm < speryear * 1e3) {
		printf("%.1f years", tm / speryear / 1e0);
	} else if (tm < speryear * 1e6) {
		printf("%.1f thousand years", tm / speryear / 1e3);
	} else if (tm < speryear * 1e9) {
		printf("%.1f million years", tm / speryear / 1e6);
	} else {
		printf("%.1f billion years", tm / speryear / 1e9);
	}
}

void saha(double rho, double T, double &H, double &Hp, double &He, double &Hep, double &Hepp, double &ne) {
	using namespace constants;
	constexpr double eps_1_H = 13.59844 * evtoerg;
	constexpr double eps_1_He = 24.58738 * evtoerg;
	constexpr double eps_2_He = 54.41776 * evtoerg;
	constexpr double g0_H = 2.0;
	constexpr double g1_H = 1.0;
	constexpr double g0_He = 1.0;
	constexpr double g1_He = 2.0;
	constexpr double g2_He = 1.0;
	const double lambda3 = pow(h * h / (2 * me * kb * T), 1.5);
	const double A0 = 2.0 / lambda3 * exp(-eps_1_H / (kb * T)) * g1_H / g0_H;
	const double B0 = 2.0 / lambda3 * exp(-eps_1_He / (kb * T)) * g1_He / g0_He;
	const double C0 = 2.0 / lambda3 * exp(-(eps_2_He - eps_1_He) / (kb * T)) * g2_He / g1_He;

	const double n_nuc = rho / mh;

	double H0 = n_nuc * (1 - global().opts.Y);
	double He0 = n_nuc * global().opts.Y / 4;
	double err;
	ne = 0.5 * (H0 + 2 * He0);
	int iters = 0;
	do {
		const double ne0 = ne;
		const double dne = -((ne - (A0 * H0) / (A0 + ne) - (B0 * He0 * (2 * C0 + ne)) / (pow(ne, 2) + B0 * (C0 + ne)))
				/ (1 + (A0 * H0) / pow(A0 + ne, 2)
						+ (B0 * He0 * (B0 * C0 + ne * (4 * C0 + ne))) / pow(pow(ne, 2) + B0 * (C0 + ne), 2)));
		ne = std::min(std::max(0.5 * ne, ne + dne), 2 * ne);
		if (ne < 1e-100) {
			break;
		}
		err = abs(log(ne / ne0));
		iters++;
		if (iters > 1000) {
			printf("Max iters exceed in compute_electron_fraction\n");
			return;
		}
	} while (err > 1.0e-6);
	Hp = std::max(A0 * H0 / (A0 + ne), 0.0);
	H = std::max(H0 - Hp, 0.0);
	Hep = std::max(B0 * ne * He0 / (B0 * C0 + B0 * ne + ne * ne), 0.0);
	Hepp = std::max(B0 * C0 * He0 / (B0 * C0 + B0 * ne + ne * ne), 0.0);
	He = std::max(He0 - Hep - Hepp, 0.0);
}

double find_root(std::function<double(double)> f) {
	double x = 0.5;
	double err;
	int iters = 0;
	do {
		double dx0 = x * 1.0e-6;
		if (abs(dx0) == 0.0) {
			dx0 = 1.0e-10;
		}
		double fx = f(x);
		double dfdx = (f(x + dx0) - fx) / dx0;
		double dx = -fx / dfdx;
		err = std::abs(dx / std::max(1.0, std::abs(x)));
		x += 0.5 * dx;
		iters++;
		if (iters > 100000) {
			printf("Finished early with error = %e\n", err);
			break;
		}
	} while (err > 1.0e-12);
	return x;
}
void chemistry_update(const std::function<double(double)> &Hubble, double &H, double &Hp, double &He, double &Hep,
		double &Hepp, double &ne, double T, double a, double dt) {
	using namespace constants;
	bool use_saha;
	double H1 = H;
	double Hp1 = Hp;
	double He1 = He;
	double Hep1 = Hep;
	double Hepp1 = Hepp;
	double ne1 = ne;
	double H0 = H;
	double Hp0 = Hp;
	double He0 = He;
	double Hep0 = Hep;
	double Hepp0 = Hepp;
	double rho = ((H + Hp) + (He + Hep + Hepp) * 4) * mh;
	if (ne > (H + Hp)) {
		saha(rho, T, H1, Hp1, He1, Hep1, Hepp1, ne1);
		if (ne1 > (H1 + Hp1)) {
			use_saha = true;
		} else {
			use_saha = false;
		}
		use_saha = true;
	} else {
		use_saha = false;
	}
	if (use_saha) {
		H = H1;
		Hp = Hp1;
		He = He1;
		Hep = Hep1;
		Hepp = Hepp1;
		ne = ne1;
	} else {
		double nH = H + Hp;
		double x0 = Hp / nH;
		const auto dxdt =
				[=](double x0, double dt) {
					double hubble = Hubble(a);
					using namespace constants;
					const double B1 = 13.6 * evtoerg;
					const double phi2 = std::max(0.448 * log(B1 / (kb * T)), 0.0);
					const double alpha2 = 64.0 * M_PI / sqrt(27.0 * M_PI) * B1 * 2.0 * pow(hbar, 2) / pow(me * c, 3) / sqrt(kb * T / B1) * phi2;
					const double beta = pow((me * kb * T) / (2 * M_PI * hbar * hbar), 1.5) * exp(-B1 / (kb * T)) * alpha2;
					const double lambda_a = 8.0 * M_PI * hbar * c / (3.0 * B1);
					const double num = h * c / lambda_a / kb / T;
					const double beta2 = beta * std::exp(std::min(num, 80.0));
					const double La = 8.0 * M_PI * hubble / (a * std::pow(lambda_a, 3) * nH);
					const double L2s = 8.227;
					const auto func = [=](double x) {
						return (x - (dt * (L2s + La / (1 - x)) * (beta * (1 - x) - alpha2 * nH * pow(x, 2))) / (beta2 + L2s + La / (1 - x)) - x0) * (1 - x);
					};
					double x = find_root(func);
					return (x - x0) / dt;
				};
		double gam = 1.0 - 1.0 / sqrt(2.0);
		double dx1 = dxdt(x0, gam * dt);
		double dx2 = dxdt(x0 + (1 - 2 * gam) * dx1 * dt, gam * dt);
		double x = (x0 + 0.5 * (dx1 * dt + dx2 * dt));
		He = He0 + Hep0 + Hepp0;
		Hep = 0.0;
		Hepp = 0.0;
		H = (1.0 - x) * (H0 + Hp0);
		Hp = x * (H0 + Hp0);
		ne = Hp;
	}
}

void create_zero_order_universe(zero_order_universe* uni_ptr, double amax);

class cosmos {
	double a;
	double t;
	double tau;
	double omega_m;
	double omega_r;
	double omega_lam;
	double H;
public:
	cosmos(double omega_c, double omega_b, double omega_gam, double omega_nu, double h_, double a_) {
		H = constants::H0 * h_;
		omega_m = omega_c + omega_b;
		omega_r = omega_gam + omega_nu;
		omega_lam = 1.0 - omega_r - omega_m;
		a = a_;
		t = tau = 0.0;
	}
	double advance(double dtau0) {
		const double beta[3] = { 1.0, 0.25, 2.0 / 3.0 };

		double tau1 = tau + dtau0;
		while (tau < tau1) {
			double a0 = a;
			double t0 = t;
			double dtau = std::min(
					abs(1.0e-2 / a / (H * sqrt(omega_m / (a * a * a) + omega_r / (a * a * a * a) + omega_lam))) * 0.01,
					tau1 - tau);
			for (int rk = 0; rk < 3; rk++) {
				double da = (H * sqrt(omega_m / (a * a * a) + omega_r / (a * a * a * a) + omega_lam)) * a * a * dtau;
				double dt = a * dtau;
				a = (1.0 - beta[rk]) * a0 + beta[rk] * (a + da);
				t = (1.0 - beta[rk]) * t0 + beta[rk] * (t + dt);
			}
			tau += dtau;
		}
		return t;
	}
	double scale() const {
		return a;
	}
	double time() const {
		return t;
	}

};

void zero_order_universe::compute_matter_fractions(float& Oc, float& Ob, float a) const {
	float omega_m = global().opts.omega_b + global().opts.omega_c;
	float omega_r = global().opts.omega_gam + global().opts.omega_nu;
	float Om = omega_m / (omega_r / a + omega_m + (a * a * a) * ((float) 1.0 - omega_m - omega_r));
	Ob = global().opts.omega_b * Om / omega_m;
	Oc = global().opts.omega_c * Om / omega_m;
}

void zero_order_universe::compute_radiation_fractions(float& Ogam, float& Onu, float a) const {
	float omega_m = global().opts.omega_b + global().opts.omega_c;
	float omega_r = global().opts.omega_gam + global().opts.omega_nu;
	float Or = omega_r / (omega_r + a * omega_m + (a * a * a * a) * ((float) 1.0 - omega_m - omega_r));
	Ogam = global().opts.omega_gam * Or / omega_r;
	Onu = global().opts.omega_nu * Or / omega_r;
}

float zero_order_universe::conformal_time_to_scale_factor(float taumax) {
	taumax *= constants::H0 / cosmic_constants::H0;
	float dlogtau = 1.0e-3;
	float a = amin;
	float logtaumax = std::log(taumax);
	float logtaumin = std::log(1.f / (a * hubble(a)));
	int N = (logtaumax - logtaumin) / dlogtau + 1;
	dlogtau = (logtaumax - logtaumin) / N;
	for (int i = 0; i < N; i++) {
		float logtau = logtaumin + (float) i * dlogtau;
		float tau = std::exp(logtau);
		float a0 = a;
		a += tau * a * a * hubble(a) * dlogtau;
		logtau = logtaumin + (float) (i + 1) * dlogtau;
		tau = std::exp(logtau);
		a = 0.75f * a0 + 0.25f * (a + tau * a * a * hubble(a) * dlogtau);
		logtau = logtaumin + ((float) i + 0.5f) * dlogtau;
		tau = std::exp(logtau);
		a = 1.f / 3.f * a0 + 2.f / 3.f * (a + tau * a * a * hubble(a) * dlogtau);
	}
	return a;
}

double zero_order_universe::redshift_to_density(double z) const {
	const double a = 1.0 / (1.0 + z);
	const double omega_m = global().opts.omega_b + global().opts.omega_c;
	const double omega_r = global().opts.omega_nu + global().opts.omega_gam;
	const double omega_l = 1.0 - omega_m - omega_r;
	const double H2 = sqr(global().opts.hubble * constants::H0)
			* (omega_r / (a * a * a * a) + omega_m / (a * a * a) + omega_l);
	return omega_m * 3.0 * H2 / (8.0 * M_PI * constants::G);
}

float zero_order_universe::scale_factor_to_conformal_time(float a) {
	float amax = a;
	float dloga = 1e-2;
	float logamin = std::log(amin);
	float logamax = std::log(amax);
	int N = (logamax - logamin) / dloga + 1;
	dloga = (logamax - logamin) / (float) N;
	float tau = 1.f / (amin * hubble(amin));
	for (int i = 0; i < N; i++) {
		float loga = logamin + (float) i * dloga;
		float a = std::exp(loga);
		float tau0 = tau;
		tau += dloga / (a * hubble(a));
		loga = logamin + (float) (i + 1) * dloga;
		a = std::exp(loga);
		tau = 0.75f * tau0 + 0.25f * (tau + dloga / (a * hubble(a)));
		loga = logamin + ((float) i + 0.5f) * dloga;
		a = std::exp(loga);
		tau = (1.f / 3.f) * tau0 + (2.f / 3.f) * (tau + dloga / (a * hubble(a)));
	}
	tau *= cosmic_constants::H0 / constants::H0;
	return tau;
}

float zero_order_universe::redshift_to_time(float z) const {
	float amax = 1.f / (1.f + z);
	float dloga = 1e-3;
	float logamin = std::log(amin);
	float logamax = std::log(amax);
	int N = (logamax - logamin) / dloga + 1;
	dloga = (logamax - logamin) / (float) N;
	float t = 0.0;
	for (int i = 0; i < N; i++) {
		float loga = logamin + (float) i * dloga;
		float a = std::exp(loga);
		float t0 = t;
		t += dloga / hubble(a);
		loga = logamin + (float) (i + 1) * dloga;
		a = std::exp(loga);
		t = 0.75f * t0 + 0.25f * (t + dloga / hubble(a));
		loga = logamin + ((float) i + 0.5f) * dloga;
		a = std::exp(loga);
		t = (1.f / 3.f) * t0 + (2.f / 3.f) * (t + dloga / hubble(a));
	}
	t *= cosmic_constants::H0 / constants::H0;
	return t;
}

void create_zero_order_universe(zero_order_universe* uni_ptr, double amax) {
	zero_order_universe& uni = *uni_ptr;
	;
	using namespace constants;
	double omega_b = global().opts.omega_b;
	double omega_c = global().opts.omega_c;
	double omega_gam = global().opts.omega_gam;
	double omega_nu = global().opts.omega_nu;
	double omega_m = omega_b + omega_c;
	double omega_r = omega_gam + omega_nu;
	double Theta = global().opts.Theta;
	double littleh = global().opts.hubble;
	double Neff = global().opts.Neff;
	double Y = global().opts.Y;
	double amin = Theta * Tcmb / (0.07 * 1e6 * evtoK);
	double logamin = log(amin);
	double logamax = log(amax);
	int N = 64 * 1024;
	double dloga = (logamax - logamin) / N;
	vector<float> thomson(N + 1);
	vector<float> sound_speed2(N + 1);

	printf("\t\tParameters:\n");
	printf("\t\t\t h                 = %f\n", littleh);
	printf("\t\t\t omega_m           = %f\n", omega_m);
	printf("\t\t\t omega_r           = %f\n", omega_r);
	printf("\t\t\t omega_lambda      = %f\n", 1 - omega_r - omega_m);
	printf("\t\t\t omega_b           = %f\n", omega_b);
	printf("\t\t\t omega_c           = %f\n", omega_c);
	printf("\t\t\t omega_gam         = %f\n", omega_gam);
	printf("\t\t\t omega_nu          = %f\n", omega_nu);
	printf("\t\t\t Neff              = %f\n", Neff);
	printf("\t\t\t temperature today = %f\n\n", 2.73 * Theta);

	dloga = (logamax - logamin) / N;
	const auto cosmic_hubble =
			[=](double a) {
				using namespace cosmic_constants;
				return littleh * cosmic_constants::H0 * sqrt(omega_r / (a * a * a * a) + omega_m / (a * a * a) + (1 - omega_r - omega_m));
			};
	auto cgs_hubble = [=](double a) {
		return constants::H0 / cosmic_constants::H0 * cosmic_hubble(a);
	};

	const auto rho_baryon = [=](double a) {
		using namespace constants;
		return 3.0 * pow(littleh * H0, 2) / (8.0 * M_PI * G) * omega_b / (a * a * a);
	};

	const auto T_radiation = [=](double a) {
		using namespace constants;
		return Tcmb * Theta / a;
	};

	double loga;
	double a;

	double rho_b, nH, nHp, nHe, nHep, nHepp, ne, Tgas, Trad;
	double hubble = cgs_hubble(amin);

	rho_b = rho_baryon(amin);
	double nnuc = rho_b / mh;
	nHepp = Y * nnuc / 4;
	nHp = (1 - Y) * nnuc;
	nH = nHe = nHep = 0.0;
	ne = nHp + 2 * nHepp;
	Trad = T_radiation(amin);
	Tgas = Trad;
	double n = nH + nHp + nHe + nHep + nHepp;
	double P = kb * (n + ne) * Tgas;
	double t = 0.0;
	double dt;
	for (int i = -10 / dloga; i <= 0; i++) {
		loga = logamin + i * dloga;
		dt = dloga / cgs_hubble(std::exp(loga));
		t += dt;
	}
	a = std::exp(logamin);
	printf("\n");
//	print_time(t);
//	printf(
//			", redshift %.0f: Big Bang nucleosynthesis has ended. The Universe is dominated by radiation at a temperature of %8.2e K."
//					" \n   Its total matter density is %.1f \% times the density of air at sea level.\n", 1 / a - 1,
//			Trad, 100 * rho_b * omega_m / omega_b / 1.274e-3);
	double mu = (nH + nHp + 4 * nHe + 4 * nHep + 4 * nHepp) * mh / (nH + nHp + nHe + nHep + nHepp + ne);
	double sigmaC = mu / me * c * (8.0 / 3.0) * omega_gam / (a * omega_m) * sigma_T * ne / hubble;
	double sigmaT = c * sigma_T * ne / hubble;
	double Hionratio = nH != 0.0 ? nHp / nH : 1e+3;
	double Heionratio = nHe != 0.0 ? (nHep + nHepp) / nHe : 1e+3;
	thomson[0] = sigmaT;
	double P1, P2;
	double rho1, rho2;
	double cs2;
	P1 = P2 = P;
	rho1 = rho2 = rho_b;
	for (int i = 1; i <= N; i++) {
		loga = logamin + i * dloga;
		a = std::exp(loga);
//		printf("%e %e %e %e %e %e\n", a, nH, nHp, nHe, nHep, nHepp);
		P2 = P1;
		P1 = P;
		rho2 = rho1;
		rho1 = rho_b;
		hubble = cgs_hubble(a);
		nH /= rho_b;
		nHp /= rho_b;
		nHe /= rho_b;
		nHep /= rho_b;
		nHepp /= rho_b;
		ne /= rho_b;
		rho_b = rho_baryon(a);
		nH *= rho_b;
		nHp *= rho_b;
		nHe *= rho_b;
		nHep *= rho_b;
		nHepp *= rho_b;
		ne *= rho_b;
		Trad = T_radiation(a);
		double dt = dloga / hubble;
		const double gamma = 1.0 - 1.0 / sqrt(2.0);
		chemistry_update(cgs_hubble, nH, nHp, nHe, nHep, nHepp, ne, Tgas, a, 0.5 * dt);
		mu = (nH + nHp + 4 * nHe + 4 * nHep + 4 * nHepp) * mh / (nH + nHp + nHe + nHep + nHepp + ne);
		sigmaC = mu / me * c * (8.0 / 3.0) * omega_gam / (a * omega_m) * sigma_T * ne / hubble;
		const double dTgasdT1 = ((Tgas + gamma * dloga * sigmaC * Trad) / (1 + gamma * dloga * (2 + sigmaC)) - Tgas)
				/ (gamma * dloga);
		const double T1 = Tgas + (1 - 2 * gamma) * dTgasdT1 * dloga;
		const double dTgasdT2 = ((T1 + gamma * dloga * sigmaC * Trad) / (1 + gamma * dloga * (2 + sigmaC)) - T1)
				/ (gamma * dloga);
		Tgas += 0.5 * (dTgasdT1 + dTgasdT2) * dloga;
		chemistry_update(cgs_hubble, nH, nHp, nHe, nHep, nHepp, ne, Tgas, a, 0.5 * dt);
		n = nH + nHp + nHe + nHep + nHepp;
		P = kb * (n + ne) * Tgas;
		sigmaT = c * sigma_T * ne / hubble;
		Hionratio = nH != 0.0 ? nHp / nH : 1e+3;
		Heionratio = nHe != 0.0 ? (nHep + nHepp) / nHe : 1e+3;
		t += dt;
		if (i == 1) {
			cs2 = (P - P1) / (rho_b - rho1);
		} else {
			cs2 = (P - P2) / (rho_b - rho2);
		}
		sound_speed2[i - 1] = cs2 / (c * c);
		thomson[i] = sigmaT;
		printf("%e %e %e %e\n", 1.0/a-1.0, (nHp + nHep + 2 * nHepp) / (nH + nHp + 2 * (nHe + nHep + nHepp)), thomson[i],
				sqrt(sound_speed2[i - 1]));
	}
	cs2 = (P - P1) / (rho_b - rho1);
	sound_speed2[N - 1] = cs2 / c;
//	print_time(t);
	uni.amin = amin;
	uni.amax = amax;
	build_interpolation_function(&uni.sigma_T, thomson, (float) amin, (float) amax);
	build_interpolation_function(&uni.cs2, sound_speed2, (float) amin, (float) amax);
	uni.hubble = std::move(cosmic_hubble);
}
