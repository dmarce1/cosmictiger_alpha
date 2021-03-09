/*
 * interactions.hpp
 *
 *  Created on: Feb 6, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_INTERACTIONS_HPP_
#define COSMICTIGER_INTERACTIONS_HPP_

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/expansion.hpp>

#ifndef __CUDA_ARCH__
template<class T>
CUDA_EXPORT inline T fma(T a, T b, T c) {
	return fmaf(a, b, c);
}
#endif

#include <cosmictiger/simd.hpp>
#include <cosmictiger/ewald_indices.hpp>

//#define EWALD_DOUBLE_PRECISION

template<class T>
CUDA_EXPORT int inline green_deriv_direct(expansion<T> &D, const T &d0, const T &d1, const T &d2, const T &d3,
		const T &d4, const array<T, NDIM> &dx);

#ifdef __CUDA_ARCH__
#define  GREEN_MAX fmaxf
#else
#define  GREEN_MAX max
#endif

template<class T>
CUDA_EXPORT inline int green_direct(expansion<T> &D, const array<T, NDIM> &dX) {
	const T r02 = T(1.0e-20);
// const T H = options::get().soft_len;
	const T nthree(-3.0f);
	const T nfive(-5.0f);
	const T nseven(-7.0f);
	const T r2 = fma(dX[0], dX[0], fma(dX[1], dX[1], sqr(dX[2])));            // 5
#ifdef __CUDA_ARCH__
			const T rinv = (r2 > r02) * rsqrtf(fmaxf(r2,r02));                  // FLOP_RSQRT + 3
#else
	const T rinv = (r2 > r02) * rsqrt(max(r2, r02));
#endif
	const T r2inv = rinv * rinv;        // 1
	const T d0 = -rinv;                 // 1
	const T d1 = -d0 * r2inv;           // 2
	const T d2 = nthree * d1 * r2inv;      // 2
	const T d3 = nfive * d2 * r2inv;    // 2
	const T d4 = nseven * d3 * r2inv;      // 2
	return 18 + FLOP_RSQRT + green_deriv_direct(D, d0, d1, d2, d3, d4, dX);
}

template<class T>
CUDA_EXPORT inline int green_deriv_ewald(expansion<T> &D, const T &d0, const T &d1, const T &d2, const T &d3,
		const T &d4, const array<T, NDIM> &dx) {
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
	D[1] = fma(dx[0], d1, D[1]);               // 2
	D[4] = fma(dx0dx0, d2, D[4]);              // 2
	dxadxbdxc = dx0dx0 * dx[0];                // 1
	D[10] = fma(dxadxbdxc, d3, D[10]);         // 2
	D[20] = fma(dxadxbdxc * dx[0], d4, D[20]); // 3
	D[2] = fma(dx[1], d1, D[2]);               // 2
	D[5] = fma(dx1dx0, d2, D[5]);              // 2
	dxadxbdxc = dx1dx0 * dx[0];                // 1
	D[11] = fma(dxadxbdxc, d3, D[11]);         // 2
	D[21] = fma(dxadxbdxc * dx[0], d4, D[21]); // 3
	D[7] = fma(dx1dx1, d2, D[7]);              // 2
	dxadxbdxc = dx1dx1 * dx[0];                // 1
	D[13] = fma(dxadxbdxc, d3, D[13]);         // 2
	D[23] = fma(dxadxbdxc * dx[0], d4, D[23]); // 3
	dxadxbdxc = dx1dx1 * dx[1];                // 1
	D[16] = fma(dxadxbdxc, d3, D[16]);         // 2
	D[26] = fma(dxadxbdxc * dx[0], d4, D[26]); // 3
	D[30] = fma(dxadxbdxc * dx[1], d4, D[30]); // 3
	D[3] = fma(dx[2], d1, D[3]);               // 2
	D[6] = fma(dx2dx0, d2, D[6]);              // 2
	dxadxbdxc = dx2dx0 * dx[0];                // 1
	D[12] = fma(dxadxbdxc, d3, D[12]);         // 2
	D[22] = fma(dxadxbdxc * dx[0], d4, D[22]); // 3
	D[8] = fma(dx2dx1, d2, D[8]);              // 2
	dxadxbdxc = dx2dx1 * dx[0];                // 1
	D[14] = fma(dxadxbdxc, d3, D[14]);         // 2
	D[24] = fma(dxadxbdxc * dx[0], d4, D[24]); // 2
	dxadxbdxc = dx2dx1 * dx[1];                // 1
	D[17] = fma(dxadxbdxc, d3, D[17]);         // 2
	D[27] = fma(dxadxbdxc * dx[0], d4, D[27]); // 3
	D[31] = fma(dxadxbdxc * dx[1], d4, D[31]); // 3
	D[9] = fma(dx2dx2, d2, D[9]);              // 2
	dxadxbdxc = dx2dx2 * dx[0];                // 1
	D[15] = fma(dxadxbdxc, d3, D[15]);         // 2
	D[25] = fma(dxadxbdxc * dx[0], d4, D[25]); // 3
	dxadxbdxc = dx2dx2 * dx[1];                // 1
	D[18] = fma(dxadxbdxc, d3, D[18]);         // 2
	D[28] = fma(dxadxbdxc * dx[0], d4, D[28]); // 2
	D[32] = fma(dxadxbdxc * dx[1], d4, D[32]); // 2
	dxadxbdxc = dx2dx2 * dx[2];                // 1
	D[19] = fma(dxadxbdxc, d3, D[19]);         // 2
	D[29] = fma(dxadxbdxc * dx[0], d4, D[29]); // 3
	D[33] = fma(dxadxbdxc * dx[1], d4, D[33]); // 3
	D[34] = fma(dxadxbdxc * dx[2], d4, D[34]); // 3

	const auto dx0d2 = dx[0] * d2;             // 1
	const auto dx1d2 = dx[1] * d2;             // 1
	const auto dx2d2 = dx[2] * d2;             // 1
	D[4] += d1;                                // 1
	D[10] = fma(T(3), dx0d2, D[10]);           // 2
	D[20] = fma(T(6) * dx0dx0, d3, D[20]);     // 3
	D[20] = fma(T(2), d2, D[20]);              // 2
	D[20] += d2;                               // 1
	D[7] += d1;                                // 1
	D[16] = fma(T(3), dx1d2, D[16]);           // 2
	D[30] = fma(T(6) * dx1dx1, d3, D[30]);     // 3
	D[30] = fma(T(2), d2, D[30]);              // 2
	D[30] += d2;                               // 1
	threedxadxb = T(3) * dx1dx0;               // 1
	D[13] += dx0d2;                            // 1
	D[11] += dx1d2;                            // 1
	D[26] = fma(threedxadxb, d3, D[26]);       // 2
	D[21] = fma(threedxadxb, d3, D[21]);       // 2
	D[23] += d2;                               // 1
	D[23] = fma(dx0dx0, d3, D[23]);            // 2
	D[23] = fma(dx1dx1, d3, D[23]);            // 2
	D[9] += d1;                                // 1
	D[19] = fma(T(3), dx2d2, D[19]);           // 2
	D[34] = fma(T(6) * dx2dx2, d3, D[34]);     // 3
	D[34] = fma(T(2), d2, D[34]);              // 2
	D[34] += d2;                               // 1
	threedxadxb = T(3) * dx2dx0;               // 1
	D[15] += dx0d2;                            // 1
	D[12] += dx2d2;                            // 1
	D[29] = fma(threedxadxb, d3, D[29]);       // 2
	D[22] = fma(threedxadxb, d3, D[22]);       // 2
	D[25] += d2;                               // 1
	D[25] = fma(dx0dx0, d3, D[25]);            // 2
	D[25] = fma(dx2dx2, d3, D[25]);            // 2
	threedxadxb = T(3) * dx2dx1;               // 1
	D[18] += dx1d2;                            // 1
	D[17] += dx2d2;                            // 1
	D[33] = fma(threedxadxb, d3, D[33]);       // 2
	D[31] = fma(threedxadxb, d3, D[31]);       // 2
	D[32] += d2;                               // 1
	D[32] = fma(dx1dx1, d3, D[32]);            // 2
	D[32] = fma(dx2dx2, d3, D[32]);            // 2
	D[28] = fma(dx1dx0, d3, D[28]);            // 2
	D[24] = fma(dx2dx1, d3, D[24]);            // 2
	D[27] = fma(dx2dx0, d3, D[27]);            // 2
	return 169;
}

#include <cuda_runtime.h>

CUDA_DEVICE inline float erfcexp(const float &x, float *e) {				// 18 + FLOP_DIV + FLOP_EXP
	const float p(0.3275911f);
	const float a1(0.254829592f);
	const float a2(-0.284496736f);
	const float a3(1.421413741f);
	const float a4(-1.453152027f);
	const float a5(1.061405429f);
	const float t1 = 1.f / fma(p, x, 1.f);			            // FLOP_DIV + 2
	const float t2 = t1 * t1;											// 1
	const float t3 = t2 * t1;											// 1
	const float t4 = t2 * t2;											// 1
	const float t5 = t2 * t3;											// 1
	*e = expf(-x * x);												  // 2 + FLOP_EXP
	return fma(a1, t1, fma(a2, t2, fma(a3, t3, fma(a4, t4, a5 * t5)))) * *e; 			// 10
}


template<class T>
CUDA_EXPORT inline int green_ewald(expansion<T> &D, const array<T, NDIM> &X) {
	const ewald_data indices;
	const T fouroversqrtpi(4.0 / sqrt(M_PI));
	const T one(1.0);
	const T nthree(-3.0);
	const T nfive(-5.0);
	const T nseven(-7.0);
	const T neight(-8.0);
	const T rcut(1.0e-6);
	const T r = sqrt(fma(X[0], X[0], fma(X[1], X[1], sqr(X[2]))));                   // 5
	const T zmask = r > rcut;                                                        // 1
	int flops = 6;
	D = 0.0;
	const auto nreal = indices.nreal();
	for (int i = 0; i < nreal; i++) {
		const auto* n = indices.real_index(i);
		array<T, NDIM> dx;
		for (int dim = 0; dim < NDIM; dim++) {                                        // 3
			dx[dim] = X[dim] - n[dim];
		}
		const T r2 = fma(dx[0], dx[0], fma(dx[1], dx[1], sqr(dx[2])));                // 5
		flops += 9;
#ifdef __CUDA_ARCH__
				if (r2 < (EWALD_REAL_CUTOFF2)) {                                        // 1
#endif
		const T r = sqrt(r2);                                                         // FLOP_SQRT
		const T cmask = one - (fma(n[0], n[0], fma(n[1], n[1], sqr(n[2]))) > 0.0);    // 7
		const T mask = (one - (one - zmask) * cmask);                                 // 3
		const T rinv = mask / max(r, rcut);                                           // 1 + FLOP_DIV
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
		const T d1 = fma(-d0, r2inv, e1);                                             // 3
		const T d2 = fma(nthree * d1, r2inv, e2);                                     // 3
		const T d3 = fma(nfive * d2, r2inv, e3);                                      // 3
		const T d4 = fma(nseven * d3, r2inv, e4);                                     // 3
		flops += 51 + 2 * FLOP_DIV + FLOP_SQRT + FLOP_EXP + green_deriv_ewald(D, d0, d1, d2, d3, d4, dx);
#ifdef __CUDA_ARCH__
	}
#endif
	}
	const T twopi = 2.0 * M_PI;
	const auto nfour = indices.nfour();
	for (int i = 0; i < nfour; i++) {
		const auto h = indices.four_index(i);
		const auto* hpart = indices.periodic_part(i);
		const T hdotx = fma(h[0], X[0], fma(h[1], X[1], h[2] * X[2]));  // 5
		T co;
		T so;
		sincosf(twopi * hdotx, &so, &co);                         // FLOP_SINCOS
		D[0] = fma(hpart[0], co, D[0]);                           // 2
		D[1] = fma(hpart[1], so, D[1]);                           // 2
		D[2] = fma(hpart[2], so, D[2]);                           // 2
		D[3] = fma(hpart[3], so, D[3]);                           // 2
		D[4] = fma(hpart[4], co, D[4]);                           // 2
		D[5] = fma(hpart[5], co, D[5]);                           // 2
		D[6] = fma(hpart[6], co, D[6]);                           // 2
		D[7] = fma(hpart[7], co, D[7]);                           // 2
		D[8] = fma(hpart[8], co, D[8]);                           // 2
		D[9] = fma(hpart[9], co, D[9]);                           // 2
		D[10] = fma(hpart[10], so, D[10]);                           // 2
		D[11] = fma(hpart[11], so, D[11]);                           // 2
		D[12] = fma(hpart[12], so, D[12]);                           // 2
		D[13] = fma(hpart[13], so, D[13]);                           // 2
		D[14] = fma(hpart[14], so, D[14]);                           // 2
		D[15] = fma(hpart[15], so, D[15]);                           // 2
		D[16] = fma(hpart[16], so, D[16]);                           // 2
		D[17] = fma(hpart[17], so, D[17]);                           // 2
		D[18] = fma(hpart[18], so, D[18]);                           // 2
		D[19] = fma(hpart[19], so, D[19]);                           // 2
		D[20] = fma(hpart[20], co, D[20]);                           // 2
		D[21] = fma(hpart[21], co, D[21]);                           // 2
		D[22] = fma(hpart[22], co, D[22]);                           // 2
		D[23] = fma(hpart[23], co, D[23]);                           // 2
		D[24] = fma(hpart[24], co, D[24]);                           // 2
		D[25] = fma(hpart[25], co, D[25]);                           // 2
		D[26] = fma(hpart[26], co, D[26]);                           // 2
		D[27] = fma(hpart[27], co, D[27]);                           // 2
		D[28] = fma(hpart[28], co, D[28]);                           // 2
		D[30] = fma(hpart[30], co, D[30]);                           // 2
		D[29] = fma(hpart[29], co, D[29]);                           // 2
		D[31] = fma(hpart[31], co, D[31]);                           // 2
		D[32] = fma(hpart[32], co, D[32]);                           // 2
		D[33] = fma(hpart[33], co, D[33]);                           // 2
		D[34] = fma(hpart[34], co, D[34]);                           // 2
		flops += 75 + FLOP_SINCOS;
	}
	expansion<T> D1;
	flops += green_direct(D1, X);
	D() = T(M_PI / 4.0) + D();                          // 1
	for (int i = 0; i < LP; i++) {                     // 70
		D[i] = fma(-zmask, D1[i], D[i]);
	}
	flops += 71;
	return flops;
}

template<class T>
CUDA_EXPORT int inline green_deriv_direct(expansion<T> &D, const T &d0, const T &d1, const T &d2, const T &d3,
		const T &d4, const array<T, NDIM> &dx) {
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
	D[10] = fma(T(3), dx0d2, D[10]);       // 2
	D[20] = fma(T(6) * dx0dx0, d3, D[20]); // 3
	D[20] = fma(T(2), d2, D[20]);          // 2
	D[20] += d2;                            // 1
	D[7] += d1;                             // 1
	D[16] = fma(T(3), dx1d2, D[16]);       // 2
	D[30] = fma(T(6) * dx1dx1, d3, D[30]); // 3
	D[30] = fma(T(2), d2, D[30]);          // 2
	D[30] += d2;                            // 1
	threedxadxb = T(3) * dx1dx0;            // 1
	D[13] += dx0d2;                         // 1
	D[11] += dx1d2;                         // 1
	D[26] = fma(threedxadxb, d3, D[26]);   // 2
	D[21] = fma(threedxadxb, d3, D[21]);   // 2
	D[23] += d2;                            // 1
	D[23] = fma(dx0dx0, d3, D[23]);        // 2
	D[23] = fma(dx1dx1, d3, D[23]);        // 2
	D[9] += d1;                             // 1
	D[19] = fma(T(3), dx2d2, D[19]);       // 2
	D[34] = fma(T(6) * dx2dx2, d3, D[34]); // 2
	D[34] = fma(T(2), d2, D[34]);          // 2
	D[34] += d2;                            // 1
	threedxadxb = T(3) * dx2dx0;            // 1
	D[15] += dx0d2;                         // 1
	D[12] += dx2d2;                         // 1
	D[29] = fma(threedxadxb, d3, D[29]);   // 2
	D[22] = fma(threedxadxb, d3, D[22]);   // 2
	D[25] += d2;                            // 1
	D[25] = fma(dx0dx0, d3, D[25]);        // 2
	D[25] = fma(dx2dx2, d3, D[25]);        // 2
	threedxadxb = T(3) * dx2dx1;            // 1
	D[18] += dx1d2;                         // 1
	D[17] += dx2d2;                         // 1
	D[33] = fma(threedxadxb, d3, D[33]);   // 2
	D[31] = fma(threedxadxb, d3, D[31]);   // 2
	D[32] += d2;                            // 1
	D[32] = fma(dx1dx1, d3, D[32]);        // 2
	D[32] = fma(dx2dx2, d3, D[32]);        // 2
	D[28] = fma(dx1dx0, d3, D[28]);        // 2
	D[24] = fma(dx2dx1, d3, D[24]);        // 2
	D[27] = fma(dx2dx0, d3, D[27]);        // 2
	return 135;
}

// 986 // 251936
template<class T>
CUDA_EXPORT inline int multipole_interaction(expansion<T> &L, const multipole_type<T> &M, const expansion<T>& D) { // 670/700 + 418 * NT + 50 * NFOUR
	int flops = 0;
	const auto half = (0.5f);
	const auto sixth = (1.0f / 6.0f);
	const auto halfD11 = half * D[11]; // 1
	const auto halfD12 = half * D[12]; // 1
	const auto halfD13 = half * D[13]; // 1
	const auto halfD15 = half * D[15]; // 1
	const auto halfD17 = half * D[17]; // 1
	const auto halfD18 = half * D[18]; // 1
	const auto halfD21 = half * D[21]; // 1
	const auto halfD22 = half * D[22]; // 1
	const auto halfD23 = half * D[23]; // 1
	const auto halfD24 = half * D[24]; // 1
	const auto halfD25 = half * D[25]; // 1
	const auto halfD26 = half * D[26]; // 1
	const auto halfD27 = half * D[27]; // 1
	const auto halfD28 = half * D[28]; // 1
	const auto halfD29 = half * D[29]; // 1
	const auto halfD31 = half * D[31]; // 1
	const auto halfD32 = half * D[32]; // 1
	const auto halfD33 = half * D[33]; // 1
	for (int i = 0; i < LP; i++) {
		L[i] = fma(M[0], D[i], L[i]);   // 70
	}
	L[0] = fma(M[1], D[4] * half, L[0]); // 3
	L[1] = fma(M[1], D[10] * half, L[1]);// 3
	L[2] = fma(M[1], halfD11, L[2]);     // 2
	L[3] = fma(M[1], halfD12, L[3]);     // 2
	L[4] = fma(M[1], D[20] * half, L[4]);// 3
	L[5] = fma(M[1], halfD21, L[5]);     // 2
	L[6] = fma(M[1], halfD22, L[6]);     // 2
	L[7] = fma(M[1], halfD23, L[7]);     // 2
	L[8] = fma(M[1], halfD24, L[8]);     // 2
	L[9] = fma(M[1], halfD25, L[9]);     // 2
	L[0] = fma(M[2], D[5], L[0]);        // 2
	L[1] = fma(M[2], D[11], L[1]);       // 2
	L[2] = fma(M[2], D[13], L[2]);       // 2
	L[3] = fma(M[2], D[14], L[3]);       // 2
	L[4] = fma(M[2], D[21], L[4]);       // 2
	L[5] = fma(M[2], D[23], L[5]);       // 2
	L[6] = fma(M[2], D[24], L[6]);       // 2
	L[7] = fma(M[2], D[26], L[7]);       // 2
	L[8] = fma(M[2], D[27], L[8]);       // 2
	L[9] = fma(M[2], D[28], L[9]);       // 2
	L[0] = fma(M[3], D[6], L[0]);        // 2
	L[1] = fma(M[3], D[12], L[1]);       // 2
	L[2] = fma(M[3], D[14], L[2]);       // 2
	L[3] = fma(M[3], D[15], L[3]);       // 2
	L[4] = fma(M[3], D[22], L[4]);       // 2
	L[5] = fma(M[3], D[24], L[5]);       // 2
	L[6] = fma(M[3], D[25], L[6]);       // 2
	L[7] = fma(M[3], D[27], L[7]);       // 2
	L[8] = fma(M[3], D[28], L[8]);       // 2
	L[9] = fma(M[3], D[29], L[9]);       // 2
	L[0] = fma(M[4], D[7] * half, L[0]); // 3
	L[1] = fma(M[4], halfD13, L[1]);     // 2
	L[2] = fma(M[4], D[16] * half, L[2]);// 3
	L[3] = fma(M[4], halfD17, L[3]);     // 2
	L[4] = fma(M[4], halfD23, L[4]);     // 2
	L[5] = fma(M[4], halfD26, L[5]);     // 2
	L[6] = fma(M[4], halfD27, L[6]);     // 2
	L[7] = fma(M[4], D[30] * half, L[7]);// 3
	L[8] = fma(M[4], halfD31, L[8]);     // 2
	L[9] = fma(M[4], halfD32, L[9]);     // 2
	L[0] = fma(M[5], D[8], L[0]);        // 2
	L[1] = fma(M[5], D[14], L[1]);       // 2
	L[2] = fma(M[5], D[17], L[2]);       // 2
	L[3] = fma(M[5], D[18], L[3]);       // 2
	L[4] = fma(M[5], D[24], L[4]);       // 2
	L[5] = fma(M[5], D[27], L[5]);       // 2
	L[6] = fma(M[5], D[28], L[6]);       // 2
	L[7] = fma(M[5], D[31], L[7]);       // 2
	L[8] = fma(M[5], D[32], L[8]);       // 2
	L[9] = fma(M[5], D[33], L[9]);       // 2
	L[0] = fma(M[6], D[9] * half, L[0]); // 3
	L[1] = fma(M[6], halfD15, L[1]);     // 2
	L[2] = fma(M[6], halfD18, L[2]);     // 2
	L[3] = fma(M[6], D[19] * half, L[3]);// 3
	L[4] = fma(M[6], halfD25, L[4]);     // 2
	L[5] = fma(M[6], halfD28, L[5]);     // 2
	L[6] = fma(M[6], halfD29, L[6]);     // 2
	L[7] = fma(M[6], halfD32, L[7]);     // 2
	L[8] = fma(M[6], halfD33, L[8]);     // 2
	L[9] = fma(M[6], D[34] * half, L[9]);// 3
	L[0] = fma(M[7], D[10] * sixth, L[0]);//3
	L[1] = fma(M[7], D[20] * sixth, L[1]);//3
	L[2] = fma(M[7], D[21] * sixth, L[2]);//3
	L[3] = fma(M[7], D[22] * sixth, L[3]);//3
	L[0] = fma(M[8], halfD11, L[0]);     // 2
	L[1] = fma(M[8], halfD21, L[1]);     // 2
	L[2] = fma(M[8], halfD23, L[2]);     // 2
	L[3] = fma(M[8], halfD24, L[3]);     // 2
	L[0] = fma(M[9], halfD12, L[0]);     // 2
	L[1] = fma(M[9], halfD22, L[1]);     // 2
	L[2] = fma(M[9], halfD24, L[2]);     // 2
	L[3] = fma(M[9], halfD25, L[3]);     // 2
	L[0] = fma(M[10], halfD13, L[0]);    // 2
	L[1] = fma(M[10], halfD23, L[1]);    // 2
	L[2] = fma(M[10], halfD26, L[2]);    // 2
	L[3] = fma(M[10], halfD27, L[3]);    // 2
	L[0] = fma(M[11], D[14], L[0]);      // 2
	L[1] = fma(M[11], D[24], L[1]);      // 2
	L[2] = fma(M[11], D[27], L[2]);      // 2
	L[3] = fma(M[11], D[28], L[3]);      // 2
	L[0] = fma(M[12], halfD15, L[0]);    // 2
	L[1] = fma(M[12], halfD25, L[1]);    // 2
	L[2] = fma(M[12], halfD28, L[2]);    // 2
	L[3] = fma(M[12], halfD29, L[3]);    // 2
	L[0] = fma(M[13], D[16] * sixth, L[0]);//3
	L[1] = fma(M[13], D[26] * sixth, L[1]);//3
	L[2] = fma(M[13], D[30] * sixth, L[2]);//3
	L[3] = fma(M[13], D[31] * sixth, L[3]);//3
	L[0] = fma(M[14], halfD17, L[0]);     // 2
	L[1] = fma(M[14], halfD27, L[1]);     // 2
	L[2] = fma(M[14], halfD31, L[2]);     // 2
	L[3] = fma(M[14], halfD32, L[3]);     // 2
	L[0] = fma(M[15], halfD18, L[0]);     // 2
	L[1] = fma(M[15], halfD28, L[1]);     // 2
	L[2] = fma(M[15], halfD32, L[2]);     // 2
	L[3] = fma(M[15], halfD33, L[3]);     // 2
	L[0] = fma(M[16], D[19] * sixth, L[0]);// 3
	L[1] = fma(M[16], D[29] * sixth, L[1]);// 3
	L[2] = fma(M[16], D[33] * sixth, L[2]);// 3
	L[3] = fma(M[16], D[34] * sixth, L[3]);// 3
	return flops + 309;
}

// 516 / 251466
template<class T>
CUDA_EXPORT inline int multipole_interaction(array<T, NDIM + 1> &L, const multipole_type<T> &M, const expansion<T>& D) { // 517 / 47428

	int flops = 0;
	flops += 1 + NDIM;
	const auto half = T(0.5);
	const auto sixth = T(1.0 / 6.0);
	const auto halfD11 = half * D[11];            // 1
	const auto halfD12 = half * D[12];            // 1
	const auto halfD13 = half * D[13];            // 1
	const auto halfD15 = half * D[15];            // 1
	const auto halfD17 = half * D[17];            // 1
	const auto halfD18 = half * D[18];            // 1
	const auto halfD21 = half * D[21];            // 1
	const auto halfD22 = half * D[22];            // 1
	const auto halfD23 = half * D[23];            // 1
	const auto halfD24 = half * D[24];            // 1
	const auto halfD25 = half * D[25];            // 1
	const auto halfD26 = half * D[26];            // 1
	const auto halfD27 = half * D[27];            // 1
	const auto halfD28 = half * D[28];            // 1
	const auto halfD29 = half * D[29];            // 1
	const auto halfD31 = half * D[31];            // 1
	const auto halfD32 = half * D[32];            // 1
	const auto halfD33 = half * D[33];            // 1
	for (int i = 0; i < NDIM + 1; i++) {
		L[i] = fma(M[0], D[i], L[i]);              // 8
	}
	L[0] = fma(M[1], D[4] * half, L[0]);          // 3
	L[1] = fma(M[1], D[10] * half, L[1]);         // 3
	L[2] = fma(M[1], halfD11, L[2]);              // 2
	L[3] = fma(M[1], halfD12, L[3]);              // 2
	L[0] = fma(M[2], D[5], L[0]);                 // 2
	L[1] = fma(M[2], D[11], L[1]);                // 2
	L[2] = fma(M[2], D[13], L[2]);                // 2
	L[3] = fma(M[2], D[14], L[3]);                // 2
	L[0] = fma(M[3], D[6], L[0]);                 // 2
	L[1] = fma(M[3], D[12], L[1]);                // 2
	L[2] = fma(M[3], D[14], L[2]);                // 2
	L[3] = fma(M[3], D[15], L[3]);                // 2
	L[0] = fma(M[4], D[7] * half, L[0]);          // 3
	L[1] = fma(M[4], halfD13, L[1]);              // 2
	L[2] = fma(M[4], D[16] * half, L[2]);         // 3
	L[3] = fma(M[4], halfD17, L[3]);              // 2
	L[0] = fma(M[5], D[8], L[0]);                 // 2
	L[1] = fma(M[5], D[14], L[1]);                // 2
	L[2] = fma(M[5], D[17], L[2]);                // 2
	L[3] = fma(M[5], D[18], L[3]);                // 2
	L[0] = fma(M[6], D[9] * half, L[0]);          // 3
	L[1] = fma(M[6], halfD15, L[1]);              // 2
	L[2] = fma(M[6], halfD18, L[2]);              // 2
	L[3] = fma(M[6], D[19] * half, L[3]);         // 3
	L[0] = fma(M[7], D[10] * sixth, L[0]);        // 3
	L[1] = fma(M[7], D[20] * sixth, L[1]);        // 3
	L[2] = fma(M[7], D[21] * sixth, L[2]);        // 3
	L[3] = fma(M[7], D[22] * sixth, L[3]);        // 3
	L[0] = fma(M[8], halfD11, L[0]);              // 2
	L[1] = fma(M[8], halfD21, L[1]);              // 2
	L[2] = fma(M[8], halfD23, L[2]);              // 2
	L[3] = fma(M[8], halfD24, L[3]);              // 2
	L[0] = fma(M[9], halfD12, L[0]);              // 2
	L[1] = fma(M[9], halfD22, L[1]);              // 2
	L[2] = fma(M[9], halfD24, L[2]);              // 2
	L[3] = fma(M[9], halfD25, L[3]);              // 2
	L[0] = fma(M[10], halfD13, L[0]);             // 2
	L[1] = fma(M[10], halfD23, L[1]);             // 2
	L[2] = fma(M[10], halfD26, L[2]);             // 2
	L[3] = fma(M[10], halfD27, L[3]);             // 2
	L[0] = fma(M[11], D[14], L[0]);               // 2
	L[1] = fma(M[11], D[24], L[1]);               // 2
	L[2] = fma(M[11], D[27], L[2]);               // 2
	L[3] = fma(M[11], D[28], L[3]);               // 2
	L[0] = fma(M[12], halfD15, L[0]);             // 2
	L[1] = fma(M[12], halfD25, L[1]);             // 2
	L[2] = fma(M[12], halfD28, L[2]);             // 2
	L[3] = fma(M[12], halfD29, L[3]);             // 2
	L[0] = fma(M[13], D[16] * sixth, L[0]);       // 3
	L[1] = fma(M[13], D[26] * sixth, L[1]);       // 3
	L[2] = fma(M[13], D[30] * sixth, L[2]);       // 3
	L[3] = fma(M[13], D[31] * sixth, L[3]);       // 3
	L[0] = fma(M[14], halfD17, L[0]);             // 2
	L[1] = fma(M[14], halfD27, L[1]);             // 2
	L[2] = fma(M[14], halfD31, L[2]);             // 2
	L[3] = fma(M[14], halfD32, L[3]);             // 2
	L[0] = fma(M[15], halfD18, L[0]);             // 2
	L[1] = fma(M[15], halfD28, L[1]);             // 2
	L[2] = fma(M[15], halfD32, L[2]);             // 2
	L[3] = fma(M[15], halfD33, L[3]);             // 2
	L[0] = fma(M[16], D[19] * sixth, L[0]);       // 3
	L[1] = fma(M[16], D[29] * sixth, L[1]);       // 3
	L[2] = fma(M[16], D[33] * sixth, L[2]);       // 3
	L[3] = fma(M[16], D[34] * sixth, L[3]);       // 3
	return 172;
}

template<class T>
CUDA_EXPORT inline int multipole_interaction(expansion<T> &L, const expansion<T>& D) { // 390 / 47301
	for (int i = 0; i < LP; i++) {
		L[i] += D[i];
	}
	return 35;
}

#endif /* COSMICTIGER_INTERACTIONS_HPP_ */
