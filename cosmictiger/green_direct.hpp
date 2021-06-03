#pragma once

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/expansion.hpp>


#define DSCALE 1e4
#define DSCALE2 1e8
#define DSCALE3 1e12
#define DSCALE4 1e16
#define DSCALE5 1e20
#define RCUT 1e-4
#define RCUT2 1e-8

CUDA_EXPORT inline bool any_true(bool a) {
	return a;
}

inline bool any_true(simd_float a) {
	return a.sum();
}

template<class T>
CUDA_EXPORT int green_direct(expansion<T> &D, array<T, NDIM> dX, T rmin = 0.f) {
	irregular_harmonic(D,dX[0],dX[1],dX[2]);
/*	bool scaled = false;
	T r2 = FMAX(FMA(dX[0], dX[0], FMA(dX[1], dX[1], sqr(dX[2]))), rmin * rmin);            // 5
	if (any_true(r2 < T(RCUT2))) {
		scaled = true;
		dX[0] *= T(DSCALE);
		dX[1] *= T(DSCALE);
		dX[2] *= T(DSCALE);
		r2 *= T(DSCALE * DSCALE);
	}
	const T rinv = RSQRT(r2);                  // FLOP_RSQRT + 3
	const T r2inv = rinv * rinv;        // 1
	const T d0 = -rinv;                 // 1
	const T d1 = -d0 * r2inv;           // 2
	const T d2 = T(-3) * d1 * r2inv;      // 2
	const T d3 = T(-5) * d2 * r2inv;    // 2
	const T d4 = T(-7) * d3 * r2inv;      // 2
	NAN_TEST(d0);NAN_TEST(d1);NAN_TEST(d2);NAN_TEST(d3);NAN_TEST(d4);
	int flops = 21 + FLOP_RSQRT + green_deriv_direct(D, d0, d1, d2, d3, d4, dX);
	if (scaled) {
		D[0] *= T(DSCALE);
		D[1] *= T(DSCALE2);
		D[2] *= T(DSCALE2);
		D[3] *= T(DSCALE2);
		D[4] *= T(DSCALE3);
		D[5] *= T(DSCALE3);
		D[6] *= T(DSCALE3);
		D[7] *= T(DSCALE3);
		D[8] *= T(DSCALE3);
		D[9] *= T(DSCALE3);
		D[10] *= T(DSCALE4);
		D[11] *= T(DSCALE4);
		D[12] *= T(DSCALE4);
		D[13] *= T(DSCALE4);
		D[14] *= T(DSCALE4);
		D[15] *= T(DSCALE4);
		D[16] *= T(DSCALE4);
		D[17] *= T(DSCALE4);
		D[18] *= T(DSCALE4);
		D[19] *= T(DSCALE4);
		D[20] *= T(DSCALE5);
		D[21] *= T(DSCALE5);
		D[22] *= T(DSCALE5);
		D[23] *= T(DSCALE5);
		D[24] *= T(DSCALE5);
		D[25] *= T(DSCALE5);
		D[26] *= T(DSCALE5);
		D[27] *= T(DSCALE5);
		D[28] *= T(DSCALE5);
		D[29] *= T(DSCALE5);
		D[30] *= T(DSCALE5);
		D[31] *= T(DSCALE5);
		D[32] *= T(DSCALE5);
		D[33] *= T(DSCALE5);
		D[34] *= T(DSCALE5);
		flops += 35;
	}*/
	return 0;
}
