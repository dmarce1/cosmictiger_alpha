#include <cosmictiger/math.hpp>

#include <curand_kernel.h>

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
		err = abs(dx / std::max(1.0, abs(x)));
		x += 0.5 * dx;
		iters++;
		if (iters > 1000000) {
			PRINT("Finished early with error = %e\n", err);
			break;
		}
	} while (err > 1.0e-6);
	return x;
}
