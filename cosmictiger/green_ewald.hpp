#pragma once

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/ewald_indices.hpp>
#include <cosmictiger/expansion.hpp>

template<class T>
CUDA_EXPORT int green_deriv_ewald(expansion<T> &D, const T &d0, const T &d1, const T &d2, const T &d3, const T &d4,
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
	D[0] += d0;                                // 1
	D[1] = FMA(dx[0], d1, D[1]);               // 2
	D[4] = FMA(dx0dx0, d2, D[4]);              // 2
	dxadxbdxc = dx0dx0 * dx[0];                // 1
	D[10] = FMA(dxadxbdxc, d3, D[10]);         // 2
	D[20] = FMA(dxadxbdxc * dx[0], d4, D[20]); // 3
	D[2] = FMA(dx[1], d1, D[2]);               // 2
	D[5] = FMA(dx1dx0, d2, D[5]);              // 2
	dxadxbdxc = dx1dx0 * dx[0];                // 1
	D[11] = FMA(dxadxbdxc, d3, D[11]);         // 2
	D[21] = FMA(dxadxbdxc * dx[0], d4, D[21]); // 3
	D[7] = FMA(dx1dx1, d2, D[7]);              // 2
	dxadxbdxc = dx1dx1 * dx[0];                // 1
	D[13] = FMA(dxadxbdxc, d3, D[13]);         // 2
	D[23] = FMA(dxadxbdxc * dx[0], d4, D[23]); // 3
	dxadxbdxc = dx1dx1 * dx[1];                // 1
	D[16] = FMA(dxadxbdxc, d3, D[16]);         // 2
	D[26] = FMA(dxadxbdxc * dx[0], d4, D[26]); // 3
	D[30] = FMA(dxadxbdxc * dx[1], d4, D[30]); // 3
	D[3] = FMA(dx[2], d1, D[3]);               // 2
	D[6] = FMA(dx2dx0, d2, D[6]);              // 2
	dxadxbdxc = dx2dx0 * dx[0];                // 1
	D[12] = FMA(dxadxbdxc, d3, D[12]);         // 2
	D[22] = FMA(dxadxbdxc * dx[0], d4, D[22]); // 3
	D[8] = FMA(dx2dx1, d2, D[8]);              // 2
	dxadxbdxc = dx2dx1 * dx[0];                // 1
	D[14] = FMA(dxadxbdxc, d3, D[14]);         // 2
	D[24] = FMA(dxadxbdxc * dx[0], d4, D[24]); // 2
	dxadxbdxc = dx2dx1 * dx[1];                // 1
	D[17] = FMA(dxadxbdxc, d3, D[17]);         // 2
	D[27] = FMA(dxadxbdxc * dx[0], d4, D[27]); // 3
	D[31] = FMA(dxadxbdxc * dx[1], d4, D[31]); // 3
	D[9] = FMA(dx2dx2, d2, D[9]);              // 2
	dxadxbdxc = dx2dx2 * dx[0];                // 1
	D[15] = FMA(dxadxbdxc, d3, D[15]);         // 2
	D[25] = FMA(dxadxbdxc * dx[0], d4, D[25]); // 3
	dxadxbdxc = dx2dx2 * dx[1];                // 1
	D[18] = FMA(dxadxbdxc, d3, D[18]);         // 2
	D[28] = FMA(dxadxbdxc * dx[0], d4, D[28]); // 2
	D[32] = FMA(dxadxbdxc * dx[1], d4, D[32]); // 2
	dxadxbdxc = dx2dx2 * dx[2];                // 1
	D[19] = FMA(dxadxbdxc, d3, D[19]);         // 2
	D[29] = FMA(dxadxbdxc * dx[0], d4, D[29]); // 3
	D[33] = FMA(dxadxbdxc * dx[1], d4, D[33]); // 3
	D[34] = FMA(dxadxbdxc * dx[2], d4, D[34]); // 3

	const auto dx0d2 = dx[0] * d2;             // 1
	const auto dx1d2 = dx[1] * d2;             // 1
	const auto dx2d2 = dx[2] * d2;             // 1
	D[4] += d1;                                // 1
	D[10] = FMA(T(3), dx0d2, D[10]);           // 2
	D[20] = FMA(T(6) * dx0dx0, d3, D[20]);     // 3
	D[20] = FMA(T(2), d2, D[20]);              // 2
	D[20] += d2;                               // 1
	D[7] += d1;                                // 1
	D[16] = FMA(T(3), dx1d2, D[16]);           // 2
	D[30] = FMA(T(6) * dx1dx1, d3, D[30]);     // 3
	D[30] = FMA(T(2), d2, D[30]);              // 2
	D[30] += d2;                               // 1
	threedxadxb = T(3) * dx1dx0;               // 1
	D[13] += dx0d2;                            // 1
	D[11] += dx1d2;                            // 1
	D[26] = FMA(threedxadxb, d3, D[26]);       // 2
	D[21] = FMA(threedxadxb, d3, D[21]);       // 2
	D[23] += d2;                               // 1
	D[23] = FMA(dx0dx0, d3, D[23]);            // 2
	D[23] = FMA(dx1dx1, d3, D[23]);            // 2
	D[9] += d1;                                // 1
	D[19] = FMA(T(3), dx2d2, D[19]);           // 2
	D[34] = FMA(T(6) * dx2dx2, d3, D[34]);     // 3
	D[34] = FMA(T(2), d2, D[34]);              // 2
	D[34] += d2;                               // 1
	threedxadxb = T(3) * dx2dx0;               // 1
	D[15] += dx0d2;                            // 1
	D[12] += dx2d2;                            // 1
	D[29] = FMA(threedxadxb, d3, D[29]);       // 2
	D[22] = FMA(threedxadxb, d3, D[22]);       // 2
	D[25] += d2;                               // 1
	D[25] = FMA(dx0dx0, d3, D[25]);            // 2
	D[25] = FMA(dx2dx2, d3, D[25]);            // 2
	threedxadxb = T(3) * dx2dx1;               // 1
	D[18] += dx1d2;                            // 1
	D[17] += dx2d2;                            // 1
	D[33] = FMA(threedxadxb, d3, D[33]);       // 2
	D[31] = FMA(threedxadxb, d3, D[31]);       // 2
	D[32] += d2;                               // 1
	D[32] = FMA(dx1dx1, d3, D[32]);            // 2
	D[32] = FMA(dx2dx2, d3, D[32]);            // 2
	D[28] = FMA(dx1dx0, d3, D[28]);            // 2
	D[24] = FMA(dx2dx1, d3, D[24]);            // 2
	D[27] = FMA(dx2dx0, d3, D[27]);            // 2
	return 169;
}

static float __constant__ __device__ rmin = 1.0e-2;
static float __constant__ __device__ fouroversqrtpi(2.256758334);
static float __constant__ __device__ nthree(-3.0);
static float __constant__ __device__ nfive(-5.0);
static float __constant__ __device__ nseven(-7.0);
static float __constant__ __device__ neight(-8.0);

template<class T>
CUDA_EXPORT int green_ewald(expansion<T> &D, array<T, NDIM> X) {
	ewald_const econst;
#ifndef __CUDA_ARCH__
	const T rmin = 1.0e-2;
	const T fouroversqrtpi(4.0 / SQRT(M_PI));
	const T nthree(-3.0);
	const T nfive(-5.0);
	const T nseven(-7.0);
	const T neight(-8.0);
#endif
	T r = SQRT(FMA(X[0], X[0], FMA(X[1], X[1], sqr(X[2]))));                   // 5
	r = FMAX(rmin, r);
	int flops = 6;
	D = 0.0;
	const auto realsz = econst.nreal();
	for (int i = 0; i < realsz; i++) {
		const auto n = econst.real_index(i);
		array<T, NDIM> dx;
		for (int dim = 0; dim < NDIM; dim++) {                                        // 3
			dx[dim] = X[dim] - n[dim];
		}
		T r2 = FMA(dx[0], dx[0], FMA(dx[1], dx[1], sqr(dx[2])));                // 5
		r2 = FMAX(r2, rmin * rmin);
		flops += 9;
#ifdef __CUDA_ARCH__
		if (r2 < (EWALD_REAL_CUTOFF2)) {                                        // 1
#else
		if ((r2 < (EWALD_REAL_CUTOFF2)).sum()) {
#endif
			const T r = SQRT(r2);                                                         // FLOP_SQRT
			const T rinv = 1.f / r;                                           // 1 + FLOP_DIV
			NAN_TEST(rinv);
			const T r2inv = rinv * rinv;                                                  // 1
			const T r3inv = r2inv * rinv;                                                 // 1
			T exp0;
			T erfc0 = erfcexp(2.f * r, &exp0);                                            // 18 + FLOP_DIV + FLOP_EXP
			const T expfactor = fouroversqrtpi * r * exp0;                                // 2
			const T e1 = expfactor * r3inv;                                               // 1
			const T e2 = neight * e1;                                                     // 1
			const T e3 = neight * e2;                                                     // 1
			const T e4 = neight * e3;                                                     // 1
			const T d0 = -erfc0 * rinv;                                                   // 2
			const T d1 = FMA(-d0, r2inv, e1);                                             // 3
			const T d2 = FMA(nthree * d1, r2inv, e2);                                     // 3
			const T d3 = FMA(nfive * d2, r2inv, e3);                                      // 3
			const T d4 = FMA(nseven * d3, r2inv, e4);                                     // 3
			NAN_TEST(d0);
			NAN_TEST(d1);
			NAN_TEST(d2);
			NAN_TEST(d3);
			NAN_TEST(d4);
			flops += 51 + 2 * FLOP_DIV + FLOP_SQRT + FLOP_EXP + green_deriv_ewald(D, d0, d1, d2, d3, d4, dx);
		}
	}
	const T twopi = 2.0 * M_PI;

	const auto foursz = econst.nfour();
	for (int i = 0; i < foursz; i++) {
		const auto &h = econst.four_index(i);
		const auto &hpart = econst.four_expansion(i);
		const T hdotx = FMA(h[0], X[0], FMA(h[1], X[1], h[2] * X[2]));                           // 5
		T co;
		T so;
		SINCOS(twopi * hdotx, &so, &co);                           // FLOP_SINCOS
		D[0] = FMA(hpart[0], co, D[0]);                           // 2
		D[1] = FMA(hpart[1], so, D[1]);                           // 2
		D[2] = FMA(hpart[2], so, D[2]);                           // 2
		D[3] = FMA(hpart[3], so, D[3]);                           // 2
		D[4] = FMA(hpart[4], co, D[4]);                           // 2
		D[5] = FMA(hpart[5], co, D[5]);                           // 2
		D[6] = FMA(hpart[6], co, D[6]);                           // 2
		D[7] = FMA(hpart[7], co, D[7]);                           // 2
		D[8] = FMA(hpart[8], co, D[8]);                           // 2
		D[9] = FMA(hpart[9], co, D[9]);                           // 2
		D[10] = FMA(hpart[10], so, D[10]);                           // 2
		D[11] = FMA(hpart[11], so, D[11]);                           // 2
		D[12] = FMA(hpart[12], so, D[12]);                           // 2
		D[13] = FMA(hpart[13], so, D[13]);                           // 2
		D[14] = FMA(hpart[14], so, D[14]);                           // 2
		D[15] = FMA(hpart[15], so, D[15]);                           // 2
		D[16] = FMA(hpart[16], so, D[16]);                           // 2
		D[17] = FMA(hpart[17], so, D[17]);                           // 2
		D[18] = FMA(hpart[18], so, D[18]);                           // 2
		D[19] = FMA(hpart[19], so, D[19]);                           // 2
		D[20] = FMA(hpart[20], co, D[20]);                           // 2
		D[21] = FMA(hpart[21], co, D[21]);                           // 2
		D[22] = FMA(hpart[22], co, D[22]);                           // 2
		D[23] = FMA(hpart[23], co, D[23]);                           // 2
		D[24] = FMA(hpart[24], co, D[24]);                           // 2
		D[25] = FMA(hpart[25], co, D[25]);                           // 2
		D[26] = FMA(hpart[26], co, D[26]);                           // 2
		D[27] = FMA(hpart[27], co, D[27]);                           // 2
		D[28] = FMA(hpart[28], co, D[28]);                           // 2
		D[30] = FMA(hpart[30], co, D[30]);                           // 2
		D[29] = FMA(hpart[29], co, D[29]);                           // 2
		D[31] = FMA(hpart[31], co, D[31]);                           // 2
		D[32] = FMA(hpart[32], co, D[32]);                           // 2
		D[33] = FMA(hpart[33], co, D[33]);                           // 2
		D[34] = FMA(hpart[34], co, D[34]);                           // 2
		flops += 75 + FLOP_SINCOS;
	}
	expansion<T> D1;
	flops += green_direct(D1, X, T(rmin));
	D1.scale_back();
	D() = T(M_PI / 4.0) + D();                          // 1
	for (int i = 0; i < LP; i++) {                     // 70
		D[i] -= D1[i];
	}
	/**** Account for r == 0 case ****/
	const T zero_mask = (r < 2 * rmin);
	for (int i = 0; i < LP; i++) {
		D[i] *= (T(1) - zero_mask);
	}
	D[0] = 2.837291e+00 * zero_mask + D[0] * (T(1) - zero_mask);
	D[4] = -4.188790e+00 * zero_mask + D[4] * (T(1) - zero_mask);
	D[7] = -4.188790e+00 * zero_mask + D[7] * (T(1) - zero_mask);
	D[9] = -4.188790e+00 * zero_mask + D[9] * (T(1) - zero_mask);
	D[20] = -7.42e+01 * zero_mask + D[20] * (T(1) - zero_mask);
	D[23] = 3.73e+01 * zero_mask + D[23] * (T(1) - zero_mask);
	D[25] = 3.73e+01 * zero_mask + D[25] * (T(1) - zero_mask);
	D[30] = -7.42e+01 * zero_mask + D[30] * (T(1) - zero_mask);
	D[32] = 3.73e+01 * zero_mask + D[32] * (T(1) - zero_mask);
	D[34] = -7.42e+01 * zero_mask + D[34] * (T(1) - zero_mask);
	flops += 71;
	return flops;
}
