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
	bool scaled = false;
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
	}
	return flops;
}

template<class T>
CUDA_EXPORT int green_deriv_direct(expansion<T> &D, const T &d0, const T &d1, const T &d2, const T &d3, const T &d4,
		const array<T, NDIM> &dx) {
	T threedxadxb;
	T dxadxbdxc;
	const auto dx0dx0 = dx[0] * dx[0]; // 1
	const auto dx0dx1 = dx[0] * dx[1]; // 1
	const auto dx0dx2 = dx[0] * dx[2]; // 1
	const auto dx1dx1 = dx[1] * dx[1]; // 1
	const auto dx1dx2 = dx[1] * dx[2]; // 1
	const auto dx2dx2 = dx[2] * dx[2]; // 1
	const auto &dx1dx0 = dx0dx1;
	const auto &dx2dx0 = dx0dx2;
	const auto &dx2dx1 = dx1dx2;
	D[0] = d0;
	D[1] = dx[0] * d1;                 // 1
	D[4] = dx0dx0 * d2;                // 1
	dxadxbdxc = dx0dx0 * dx[0];        // 1
	D[10] = dxadxbdxc * d3;            // 1
	D[20] = dxadxbdxc * dx[0] * d4;    // 2
	D[2] = dx[1] * d1;                 // 1
	D[5] = dx1dx0 * d2;                // 1
	dxadxbdxc = dx1dx0 * dx[0];        // 1
	D[11] = dxadxbdxc * d3;            // 1
	D[21] = dxadxbdxc * dx[0] * d4;    // 2
	D[7] = dx1dx1 * d2;                // 1
	dxadxbdxc = dx1dx1 * dx[0];        // 1
	D[13] = dxadxbdxc * d3;            // 1
	D[23] = dxadxbdxc * dx[0] * d4;    // 2
	dxadxbdxc = dx1dx1 * dx[1];        // 1
	D[16] = dxadxbdxc * d3;            // 1
	D[26] = dxadxbdxc * dx[0] * d4;    // 2
	D[30] = dxadxbdxc * dx[1] * d4;    // 2
	D[3] = dx[2] * d1;                 // 1
	D[6] = dx2dx0 * d2;                // 1
	dxadxbdxc = dx2dx0 * dx[0];        // 1
	D[12] = dxadxbdxc * d3;            // 1
	D[22] = dxadxbdxc * dx[0] * d4;    // 2
	D[8] = dx2dx1 * d2;                // 1
	dxadxbdxc = dx2dx1 * dx[0];        // 1
	D[14] = dxadxbdxc * d3;            // 1
	D[24] = dxadxbdxc * dx[0] * d4;    // 2
	dxadxbdxc = dx2dx1 * dx[1];        // 1
	D[17] = dxadxbdxc * d3;            // 1
	D[27] = dxadxbdxc * dx[0] * d4;    // 2
	D[31] = dxadxbdxc * dx[1] * d4;    // 2
	D[9] = dx2dx2 * d2;                // 1
	dxadxbdxc = dx2dx2 * dx[0];        // 1
	D[15] = dxadxbdxc * d3;            // 1
	D[25] = dxadxbdxc * dx[0] * d4;    // 2
	dxadxbdxc = dx2dx2 * dx[1];        // 1
	D[18] = dxadxbdxc * d3;            // 1
	D[28] = dxadxbdxc * dx[0] * d4;    // 2
	D[32] = dxadxbdxc * dx[1] * d4;    // 2
	dxadxbdxc = dx2dx2 * dx[2];        // 1
	D[19] = dxadxbdxc * d3;            // 1
	D[29] = dxadxbdxc * dx[0] * d4;    // 2
	D[33] = dxadxbdxc * dx[1] * d4;    // 2
	D[34] = dxadxbdxc * dx[2] * d4;    // 2

	const auto dx0d2 = dx[0] * d2;          // 1
	const auto dx1d2 = dx[1] * d2;          // 1
	const auto dx2d2 = dx[2] * d2;          // 1
	D[4] += d1;                             // 1
	D[10] = FMA(T(3), dx0d2, D[10]);       // 2
	D[20] = FMA(T(6) * dx0dx0, d3, D[20]); // 3
	D[20] = FMA(T(2), d2, D[20]);          // 2
	D[20] += d2;                            // 1
	D[7] += d1;                             // 1
	D[16] = FMA(T(3), dx1d2, D[16]);       // 2
	D[30] = FMA(T(6) * dx1dx1, d3, D[30]); // 3
	D[30] = FMA(T(2), d2, D[30]);          // 2
	D[30] += d2;                            // 1
	threedxadxb = T(3) * dx1dx0;            // 1
	D[13] += dx0d2;                         // 1
	D[11] += dx1d2;                         // 1
	D[26] = FMA(threedxadxb, d3, D[26]);   // 2
	D[21] = FMA(threedxadxb, d3, D[21]);   // 2
	D[23] += d2;                            // 1
	D[23] = FMA(dx0dx0, d3, D[23]);        // 2
	D[23] = FMA(dx1dx1, d3, D[23]);        // 2
	D[9] += d1;                             // 1
	D[19] = FMA(T(3), dx2d2, D[19]);       // 2
	D[34] = FMA(T(6) * dx2dx2, d3, D[34]); // 2
	D[34] = FMA(T(2), d2, D[34]);          // 2
	D[34] += d2;                            // 1
	threedxadxb = T(3) * dx2dx0;            // 1
	D[15] += dx0d2;                         // 1
	D[12] += dx2d2;                         // 1
	D[29] = FMA(threedxadxb, d3, D[29]);   // 2
	D[22] = FMA(threedxadxb, d3, D[22]);   // 2
	D[25] += d2;                            // 1
	D[25] = FMA(dx0dx0, d3, D[25]);        // 2
	D[25] = FMA(dx2dx2, d3, D[25]);        // 2
	threedxadxb = T(3) * dx2dx1;            // 1
	D[18] += dx1d2;                         // 1
	D[17] += dx2d2;                         // 1
	D[33] = FMA(threedxadxb, d3, D[33]);   // 2
	D[31] = FMA(threedxadxb, d3, D[31]);   // 2
	D[32] += d2;                            // 1
	D[32] = FMA(dx1dx1, d3, D[32]);        // 2
	D[32] = FMA(dx2dx2, d3, D[32]);        // 2
	D[28] = FMA(dx1dx0, d3, D[28]);        // 2
	D[24] = FMA(dx2dx1, d3, D[24]);        // 2
	D[27] = FMA(dx2dx0, d3, D[27]);        // 2
	return 135;
}
