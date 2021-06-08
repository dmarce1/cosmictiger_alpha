#pragma once

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/ewald_indices.hpp>
#include <cosmictiger/expansion.hpp>

inline float first(simd_float a) {
	return a[0];
}

CUDA_EXPORT
inline float first(float a) {
	return a;
}

template<class T>
CUDA_EXPORT int green_ewald(expansion<T> &D, array<T, NDIM> X) {
	ewald_const econst;
	T r = SQRT(FMA(X[0], X[0], FMA(X[1], X[1], sqr(X[2]))));                   // 5
#ifdef __CUDA_ARCH__
			constexpr T fouroversqrtpi = 2.256758334f;
#else
	const T fouroversqrtpi = 2.256758334f;
#endif
	int flops = 6;
	D = 0.0;
	tensor_sym<T, LORDER> Dsym;
	Dsym = 0.0;
	const auto realsz = econst.nreal();
	for (int i = 0; i < realsz; i++) {
		const auto n = econst.real_index(i);
		array<T, NDIM> dx;
		for (int dim = 0; dim < NDIM; dim++) {                                        // 3
			dx[dim] = X[dim] - n[dim];
		}
		T r2 = FMA(dx[0], dx[0], FMA(dx[1], dx[1], sqr(dx[2])));                // 5
		flops += 9;
#ifdef __CUDA_ARCH__
		if (r2 < (EWALD_REAL_CUTOFF2)) {                                        // 1
#else
		if ((r2 < (EWALD_REAL_CUTOFF2)).sum()) {
#endif
			const T r = SQRT(r2);                                                         // FLOP_SQRT
			const T rinv = (r > T(0)) / max(r, 1.0e-20);                                           // 1 + FLOP_DIV
			NAN_TEST(rinv);
			const T r2inv = rinv * rinv;                                                  // 1
			const T r3inv = r2inv * rinv;                                                 // 1
			T exp0;
			T erfc0 = erfcexp(2.f * r, &exp0);                                            // 18 + FLOP_DIV + FLOP_EXP
			const T expfactor = fouroversqrtpi * exp0;                                // 2
			T e1 = expfactor * r2inv;                                               // 1
			array<T, LORDER> d;
			d[0] = -erfc0 * rinv;
			for (int l = 1; l < LORDER; l++) {
				d[l] = FMA(T(-2 * l + 1) * d[l - 1], r2inv, e1);
				e1 *= T(-8);
			}
			array<T, LORDER> rinv2pow;
			rinv2pow[0] = T(1);                                               // 2
			for (int l = 1; l < LORDER; l++) {
				rinv2pow[l] = -rinv2pow[l - 1] * rinv * rinv;
			}
			const auto D0 = vector_to_sym_tensor<T, LORDER>(dx);
			tensor_sym<T, LORDER> D1;
			array<int, NDIM> m;
			array<int, NDIM> k;
			array<int, NDIM> n;
			for (n[0] = 0; n[0] < LORDER; n[0]++) {
				for (n[1] = 0; n[1] < LORDER - n[0]; n[1]++) {
					for (n[2] = 0; n[2] < LORDER - n[0] - n[1]; n[2]++) {
						D1(n) = T(0);
						const int n0 = n[0] + n[1] + n[2];
						for (m[0] = 0; m[0] <= n[0] / 2; m[0]++) {
							for (m[1] = 0; m[1] <= n[1] / 2; m[1]++) {
								for (m[2] = 0; m[2] <= n[2] / 2; m[2]++) {
									const int m0 = m[0] + m[1] + m[2];
									T num = T(n1pow(m0) * dfactorial(2 * n0 - 2 * m0 - 1) * vfactorial(n));
									T den = T((1 << m0) * vfactorial(m) * vfactorial(n - (m) * 2));
									const T fnm = num / den;
									for (k[0] = 0; k[0] <= m0; k[0]++) {
										for (k[1] = 0; k[1] <= m0 - k[0]; k[1]++) {
											k[2] = m0 - k[0] - k[1];
											const auto p = n - (m) * 2 + (k) * 2;
											num = factorial(m0);
											den = vfactorial(k);
											const T number = fnm * num / den;
											D1(n) += number * D0(p) * d[n0 - m0] / T(dfactorial(2 * (n0 - m0) - 1)) * rinv2pow[m0];
										}
									}
								}
							}
						}
					}
				}
			}
			Dsym = Dsym + D1;
		}
	}

	const auto foursz = econst.nfour();
	for (int i = 0; i < foursz; i++) {
		const auto &h = econst.four_index(i);
		const auto& D0 = econst.four_expansion(i);
		const T hdotx = FMA(h[0], X[0], FMA(h[1], X[1], h[2] * X[2]));                           // 5
		T co;
		T so;
		SINCOS(T(2.0f * M_PI) * hdotx, &so, &co);
		T soco[2] = { co, so };
		array<int, NDIM> k;
		for (k[0] = 0; k[0] < LORDER; k[0]++) {
			for (k[1] = 0; k[1] < LORDER - k[0]; k[1]++) {
				for (k[2] = 0; k[2] < LORDER - k[0] - k[1]; k[2]++) {
					const int k0 = k[0] + k[1] + k[2];
					Dsym(k) += soco[k0 % 2] * D0(k);
				}
			}
		}
	}
	//printf("%e %e\n", abs(first(Dsym(2, 0, 1))) + abs(first(Dsym(0, 2, 1))) + abs(first(Dsym(0, 0, 3))), first(Dsym(2, 0, 1)) + first(Dsym(0, 2, 1)) + first(Dsym(0, 0, 3)));
	D = Dsym.detraceD();
	expansion<T> D1;
	green_direct(D1, X);
	D(0, 0, 0) = T(M_PI / 4.0) + D(0, 0, 0);                          // 1
	for (int i = 0; i < LP; i++) {                     // 70
		D[i] -= D1[i];
	}
	/*D[0] = 2.837291e+00 * zero_mask + D[0] * (T(1) - zero_mask);
	 D[4] = -4.188790e+00 * zero_mask + D[4] * (T(1) - zero_mask);
	 D[7] = -4.188790e+00 * zero_mask + D[7] * (T(1) - zero_mask);
	 D[9] = -4.188790e+00 * zero_mask + D[9] * (T(1) - zero_mask);
	 D[20] = -7.42e+01 * zero_mask + D[20] * (T(1) - zero_mask);
	 D[23] = 3.73e+01 * zero_mask + D[23] * (T(1) - zero_mask);
	 D[25] = 3.73e+01 * zero_mask + D[25] * (T(1) - zero_mask);
	 D[30] = -7.42e+01 * zero_mask + D[30] * (T(1) - zero_mask);
	 D[32] = 3.73e+01 * zero_mask + D[32] * (T(1) - zero_mask);
	 D[34] = -7.42e+01 * zero_mask + D[34] * (T(1) - zero_mask);*/
	return 0;
}
