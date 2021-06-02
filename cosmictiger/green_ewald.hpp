#pragma once

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/ewald_indices.hpp>
#include <cosmictiger/expansion.hpp>


template<class T>
CUDA_EXPORT int green_ewald(expansion<T> &D, array<T, NDIM> X) {
/*	ewald_const econst;
	T r = SQRT(FMA(X[0], X[0], FMA(X[1], X[1], sqr(X[2]))));                   // 5
#ifdef __CUDA_ARCH__
	constexpr T rmin = 0.01f;
	constexpr T fouroversqrtpi = 2.256758334f;
#else
	const T rmin = 0.01f;
	const T fouroversqrtpi = 2.256758334f;
#endif
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
			const T e2 = T(-8) * e1;                                                     // 1
			const T e3 = T(-8) * e2;                                                     // 1
			const T e4 = T(-8) * e3;                                                     // 1
			const T d0 = -erfc0 * rinv;                                                   // 2
			const T d1 = FMA(-d0, r2inv, e1);                                             // 3
			const T d2 = FMA(T(-3) * d1, r2inv, e2);                                     // 3
			const T d3 = FMA(T(-5) * d2, r2inv, e3);                                      // 3
			const T d4 = FMA(T(-7) * d3, r2inv, e4);                                     // 3
			NAN_TEST(d0);NAN_TEST(d1);NAN_TEST(d2);NAN_TEST(d3);NAN_TEST(d4);
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
		for (int i = 1; i < 4; i++) {
			D[i] = FMA(hpart[i], so, D[i]);                           // 2
		}
		for (int i = 4; i < 10; i++) {
			D[i] = FMA(hpart[i], co, D[i]);                           // 2
		}
		for (int i = 10; i < 20; i++) {
			D[i] = FMA(hpart[i], so, D[i]);                           // 2
		}
		for (int i = 20; i < 35; i++) {
			D[i] = FMA(hpart[i], co, D[i]);                           // 2
		}
		flops += 75 + FLOP_SINCOS;
	}
	expansion<T> D1;
	flops += green_direct(D1, X, T(rmin));
	D() = T(M_PI / 4.0) + D();                          // 1
	for (int i = 0; i < LP; i++) {                     // 70
		D[i] -= D1[i];
	}
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
	flops += 71;*/
	return 0;
}
