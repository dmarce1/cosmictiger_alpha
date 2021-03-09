/*
 * expansion.cpp
 *
 *  Created on: Feb 10, 2021
 *      Author: dmarce1
 */

#include <cosmictiger/expansion.hpp>
#include <cosmictiger/array.hpp>

static __constant__ std::array<float,LP> Larray = { 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
		5.00000000e-01, 1.00000000e+00, 1.00000000e+00, 5.00000000e-01, 1.00000000e+00, 5.00000000e-01, 1.66666672e-01,
		5.00000000e-01, 5.00000000e-01, 5.00000000e-01, 1.00000000e+00, 5.00000000e-01, 1.66666672e-01, 5.00000000e-01,
		5.00000000e-01, 1.66666672e-01, 4.16666679e-02, 1.66666672e-01, 1.66666672e-01, 2.50000000e-01, 4.99999940e-01,
		2.50000000e-01, 1.66666672e-01, 4.99999940e-01, 4.99999940e-01, 1.66666672e-01, 4.16666679e-02, 1.66666672e-01,
		2.50000000e-01, 1.66666672e-01, 4.16666679e-02 };


CUDA_EXPORT expansion<float>& shift_expansion(expansion<float> &me, const array<float, NDIM> &dX) {
	const expansion<float>& Lfactor = reinterpret_cast<const expansion<float>&>(Larray);
	for (int a = 0; a < 3; a++) {
		me() += me(a) * dX[a];
		for (int b = 0; b <= a; b++) {
			me() += me(a, b) * dX[a] * dX[b] * Lfactor(a, b);
			for (int c = 0; c <= b; c++) {
				me() += me(a, b, c) * dX[a] * dX[b] * dX[c] * Lfactor(a, b, c);
				for (int d = 0; d <= c; d++) {
					me() += me(a, b, c, d) * dX[a] * dX[b] * dX[c] * dX[d] * Lfactor(a, b, c, d);
				}
			}
		}
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			me(a) += me(a, b) * dX[b];
			for (int c = 0; c <= b; c++) {
				me(a) += me(a, b, c) * dX[b] * dX[c] * Lfactor(b, c);
				for (int d = 0; d <= c; d++) {
					me(a) += me(a, b, c, d) * dX[b] * dX[c] * dX[d] * Lfactor(b, c, d);
				}
			}
		}
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b <= a; b++) {
			for (int c = 0; c < NDIM; c++) {
				me(a, b) += me(a, b, c) * dX[c];
				for (int d = 0; d <= c; d++) {
					me(a, b) += me(a, b, c, d) * dX[c] * dX[d] * Lfactor(c, d);
				}
			}
		}
	}

	for (int a = 0; a < 3; a++) {
		for (int b = 0; b <= a; b++) {
			for (int c = 0; c <= b; c++) {
				for (int d = 0; d < 3; d++) {
					me(a, b, c) += me(a, b, c, d) * dX[d];
				}
			}
		}
	}

	return me;
}

CUDA_EXPORT void shift_expansion(expansion<float> &me, array<float, NDIM> &g, float &phi,
		const array<float, NDIM> &dX) {
	const expansion<float>& Lfactor = reinterpret_cast<const expansion<float>&>(Larray);
	phi = me();
	for (int a = 0; a < 3; a++) {
		phi += me(a) * dX[a];
		for (int b = a; b < 3; b++) {
			phi += me(a, b) * dX[a] * dX[b] * Lfactor(a, b);
			for (int c = b; c < 3; c++) {
				phi += me(a, b, c) * dX[a] * dX[b] * dX[c] * Lfactor(a, b, c);
				for (int d = c; d < 3; d++) {
					phi += me(a, b, c, d) * dX[a] * dX[b] * dX[c] * dX[d] * Lfactor(a, b, c, d);
				}
			}
		}
	}
	for (int a = 0; a < 3; a++) {
		g[a] = -me(a);
		for (int b = 0; b < 3; b++) {
			g[a] -= me(a, b) * dX[b];
			for (int c = b; c < 3; c++) {
				g[a] -= me(a, b, c) * dX[b] * dX[c] * Lfactor(b, c);
				for (int d = c; d < 3; d++) {
					g[a] -= me(a, b, c, d) * dX[b] * dX[c] * dX[d] * Lfactor(b, c, d);
				}
			}
		}
	}
}
