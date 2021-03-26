#include <cosmictiger/map.hpp>
#include <cosmictiger/global.hpp>

#include <cosmictiger/math.hpp>

void map_coords(double& x, double& y, double lambda, double phi) {
	double theta = phi;
	double theta0;
	int iters = 0;
	do {
		theta0 = theta;
		theta -= (2.0 * theta + std::sin(2.0 * theta) - M_PI * std::sin(phi)) / (2.0 + 2.0 * std::cos(2.0 * theta));
		iters++;
	} while (std::abs(theta0 - theta) > 1.0e-6);
//	printf( "%i\n", iters);
}

void map_add_part(array<double, NDIM> Y0, array<double, NDIM> Y1, double tau, double dtau, double tau_max) {
	array<double, NDIM> x0;
	array<double, NDIM> x1;
	double tau0, tau1;
	const auto map_freq = global().opts.map_freq * tau_max;
	static int nmapped = 0;
	double x20, x21, R20, R21;
	for (int xi = -1; xi <= 0; xi++) {
		x0[0] = Y0[0] + double(xi);
		x1[0] = Y1[0] + double(xi);
		x20 = sqr(x0[0]);
		x21 = sqr(x1[0]);
		for (int yi = -1; yi <= 0; yi++) {
			x0[1] = Y0[1] + double(yi);
			x1[1] = Y1[1] + double(yi);
			R20 = sqr(x0[1]) + x20;
			R21 = sqr(x1[1]) + x21;
			for (int zi = -1; zi <= 0; zi++) {
				x0[2] = Y0[2] + double(zi);
				x1[2] = Y1[2] + double(zi);
				const auto r1 = std::sqrt(R21 + sqr(x1[2]));
				if (r1 <= 1.0) {
					const auto r0 = std::sqrt(R20 + sqr(x0[2]));
					tau0 = tau + r0;
					tau1 = tau + dtau + r1;
					const int i0 = tau0 / map_freq;
					const int i1 = tau1 / map_freq;
					for (int j = i0; j < i1; j++) {
						auto x = (x0[0] + x1[0]) * 0.5;
						auto y = (x0[1] + x1[1]) * 0.5;
						const auto z = (x0[2] + x1[2]) * 0.5;
						const auto R = std::sqrt(x * x + y * y);
						const auto r = std::sqrt(x * x + y * y + z * z);
						const auto zor = z / r;
						const auto theta = std::acos(zor);
						const auto phi = std::atan2(y, x);
						map_coords(x,y,phi,theta);
					}
				}
			}
		}
	}

}
