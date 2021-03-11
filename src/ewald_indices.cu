#include <cosmictiger/ewald_indices.hpp>
#include <cosmictiger/interactions.hpp>

#define NREAL 147
#define NFOUR 92

static array<array<float, NDIM>, NREAL> real_indices;
static array<array<float, NDIM>, NFOUR> four_indices;
static array<expansion<float>, NFOUR> four_expanse;

static __managed__ array<array<float, NDIM>, NREAL> real_indices_dev;
static __managed__ array<array<float, NDIM>, NFOUR> four_indices_dev;
static __managed__ array<expansion<float>, NFOUR> four_expanse_dev;

void ewald_const::init() {
	int n2max = 10;
	int nmax = std::sqrt(n2max) + 1;
	array<float, NDIM> this_h;
	int count = 0;
	for (int i = -nmax; i <= nmax; i++) {
		for (int j = -nmax; j <= nmax; j++) {
			for (int k = -nmax; k <= nmax; k++) {
				const int i2 = i * i + j * j + k * k;
				if (i2 <= n2max) {
					this_h[0] = i;
					this_h[1] = j;
					this_h[2] = k;
					real_indices[count++] = this_h;
				}
			}
		}
	}
	n2max = 8;
	nmax = std::sqrt(n2max) + 1;
	count = 0;
	for (int i = -nmax; i <= nmax; i++) {
		for (int j = -nmax; j <= nmax; j++) {
			for (int k = -nmax; k <= nmax; k++) {
				if (i * i + j * j + k * k <= n2max) {
					this_h[0] = i;
					this_h[1] = j;
					this_h[2] = k;
					const auto hdot = sqr(this_h[0]) + sqr(this_h[1]) + sqr(this_h[2]);
					if (hdot > 0) {
						four_indices[count++] = this_h;
					}
				}
			}
		}
	}
	printf("nfour = %i\n", count);
	count = 0;
	for (int i = 0; i < NFOUR; i++) {
		array<float, NDIM> h = four_indices[i];
		const float h2 = sqr(h[0]) + sqr(h[1]) + sqr(h[2]);                     // 5 OP
		expansion<float> D;
		D = 0.0;
		const float c0 = 1.0 / h2 * exp(-M_PI * M_PI * h2 / 4.0);
		D() = -(1.0 / M_PI) * c0;
		for (int a = 0; a < NDIM; a++) {
			D(a) = 2.0 * h[a] * c0;
			for (int b = 0; b <= a; b++) {
				D(a, b) = 4.0 * M_PI * h[a] * h[b] * c0;
				for (int c = 0; c <= b; c++) {
					D(a, b, c) = -8.0 * M_PI * M_PI * h[a] * h[b] * h[c] * c0;
					for (int d = 0; d <= c; d++) {
						D(a, b, c, d) = -16.0 * M_PI * M_PI * M_PI * h[a] * h[b] * h[c] * h[d] * c0;
					}
				}
			}
		}
		four_expanse[count++] = D;
	}
	real_indices_dev = real_indices;
	four_indices_dev = four_indices;
	four_expanse_dev = four_expanse;
/*
	const auto rmax = 5;
	const auto fmax = 3;
	expansion<double> D;
	D = 0.0;
	double dx = 1.0e-3;
	for (double rx = -dx; rx < 2 * dx; rx += 2 * dx) {
		for (double ry = -dx; ry < 2 * dx; ry += 2 * dx) {
			for (double rz = -dx; rz < 2 * dx; rz += 2 * dx) {
				for (int ix = -rmax; ix <= rmax; ix++) {
					for (int jx = -rmax; jx <= rmax; jx++) {
						for (int kx = -rmax; kx <= rmax; kx++) {
							array<double, NDIM> X;
							X[0] = ix - rx;
							X[1] = jx - ry;
							X[2] = kx - rz;
							const double r2 = sqr(X[0]) + sqr(X[1]) + sqr(X[2]);
							double r = sqrt(r2);
							double rinv = 1.f / r;                                           // 1 + FLOP_DIV
							double r2inv = rinv * rinv;                                                  // 1
							double r3inv = r2inv * rinv;                                                 // 1
							double erfc0 = erfc(2.0 * r);
							double exp0 = exp(-4.0 * r * r);
							double expfactor = (4.0 / sqrt(M_PI)) * r * exp0;                                // 2
							double e1 = expfactor * r3inv;                                               // 1
							double e2 = -8.0 * e1;                                                     // 1
							double e3 = -8.0 * e2;                                                     // 1
							double e4 = -8.0 * e3;                                                     // 1
							double d0 = -erfc0 * rinv;                                                   // 2
							double d1 = -d0 * r2inv + e1;                                             // 3
							double d2 = -3.0 * d1 * r2inv + e2;                                     // 3
							double d3 = -5.0 * d2 * r2inv + e3;                                      // 3
							double d4 = -7.0 * d3 * r2inv + e4;                                     // 3
							green_deriv_ewald(D, d0, d1, d2, d3, d4, X);
						}
					}
				}
				for (int ix = -fmax; ix <= fmax; ix++) {
					for (int jx = -fmax; jx <= fmax; jx++) {
						for (int kx = -fmax; kx <= fmax; kx++) {
							array<double, NDIM> h;
							h[0] = ix;
							h[1] = jx;
							h[2] = kx;
							double h2 = sqr(h[0]) + sqr(h[1]) + sqr(h[2]);
							if (h2 > 0.0) {
								const double c0 = 1.0 / h2 * exp(-M_PI * M_PI * h2 / 4.0);
								double h2X = h[0] * rx + h[1] * ry + h[2] * rz;
								const double co = cos(2.0 * M_PI * h2X);
								const double so = sin(2.0 * M_PI * h2X);
								D() += -(1.0 / M_PI) * c0 * co;
								for (int a = 0; a < NDIM; a++) {
									D(a) += 2.0 * h[a] * c0 * so;
									for (int b = 0; b <= a; b++) {
										D(a, b) += 4.0 * M_PI * h[a] * h[b] * c0 * co;
										for (int c = 0; c <= b; c++) {
											D(a, b, c) += -8.0 * M_PI * M_PI * h[a] * h[b] * h[c] * c0 * so;
											for (int d = 0; d <= c; d++) {
												D(a, b, c, d) += -16.0 * M_PI * M_PI * M_PI * h[a] * h[b] * h[c] * h[d] * c0 * co;
											}
										}
									}
								}
							}
						}
					}
				}
				D[0] += M_PI / 4.0;
				array<double, NDIM> r;
				r[0] = rx;
				r[1] = ry;
				r[2] = rz;
				expansion<double> D1;
				green_direct(D1, r);
				for (int i = 0; i < LP; i++) {
					D[i] -= D1[i];
				}
			}
		}
	}
	for (int i = 0; i < LP; i++) {
		D[i] /= 8.0;
		printf("%e\n", D[i]);
	}
*/
}

CUDA_EXPORT int ewald_const::nfour() {
	return NFOUR;
}

CUDA_EXPORT int ewald_const::nreal() {
	return NREAL;
}

CUDA_EXPORT const array<float, NDIM>& ewald_const::real_index(int i) {
#ifdef __CUDA_ARCH__
	return (real_indices_dev)[i];
#else
	return real_indices[i];
#endif
}

CUDA_EXPORT const array<float, NDIM>& ewald_const::four_index(int i) {
#ifdef __CUDA_ARCH__
	return (four_indices_dev)[i];
#else
	return four_indices[i];
#endif
}

CUDA_EXPORT const expansion<float>& ewald_const::four_expansion(int i) {
#ifdef __CUDA_ARCH__
	return (four_expanse_dev)[i];
#else
	return four_expanse[i];
#endif
}



/*

D[0] = 2.837291e+00
D[4] = -4.188790e+00
D[7]= -4.188790e+00+
D[9] = -4.188790e+00
D[20] = -7.42e+01
D[23] = 3.73e+01
D[25] = 3.73e+01
D[30] = -7.42e+01
D[32] = 3.73e+01
D[34] = -7.42e+01
*/
