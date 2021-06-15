#pragma once

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/ewald_indices.hpp>
#include <cosmictiger/expansion.hpp>


inline bool anytrue(simd_float b) {
	return b.sum() != 0.0;
}

CUDA_EXPORT
inline bool anytrue(float b) {
	return b != 0.0;
}

template<class T>
CUDA_EXPORT int green_ewald(expansion<T> &D, array<T, NDIM> X) {
	ewald_greens_function(D, X);
	return 0;
	ewald_const econst;
	T r = SQRT(FMA(X[0], X[0], FMA(X[1], X[1], sqr(X[2]))));                   // 5
	const T fouroversqrtpi = T(4.0 / sqrt(M_PI));
	int flops = 6;
	tensor_sym<T, LORDER> Dreal;
	expansion<T> Dfour;
	Dreal = 0.0f;
	Dfour = 0.0f;
	D = 0.0f;
	const auto realsz = econst.nreal();
	const T zero_mask = (sqr(X[0], X[1], X[2]) > T(0));
	for (int i = 0; i < realsz; i++) {
		const auto n = econst.real_index(i);
		array<T, NDIM> dx;
		for (int dim = 0; dim < NDIM; dim++) {                                        // 3
			dx[dim] = X[dim] - n[dim];
		}
		T r2 = FMA(dx[0], dx[0], FMA(dx[1], dx[1], sqr(dx[2])));                // 5
		flops += 9;
		if (anytrue(r2 < (EWALD_REAL_CUTOFF2))) {
			const T r = SQRT(r2);                                                         // FLOP_SQRT
			const T rinv = (r > T(0)) / max(r, 1.0e-20);                                           // 1 + FLOP_DIV
			NAN_TEST(rinv);
			const T r2inv = rinv * rinv;                                                  // 1
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
				rinv2pow[l] = rinv2pow[l - 1] * rinv * rinv;
			}
			const auto D0 = vector_to_sym_tensor<T, LORDER>(dx);
			array<int, NDIM> m;
			array<int, NDIM> k;
			array<int, NDIM> n;
			for (n[0] = 0; n[0] < LORDER; n[0]++) {
				for (n[1] = 0; n[1] < LORDER - n[0]; n[1]++) {
					for (n[2] = 0; n[2] < LORDER - n[0] - n[1]; n[2]++) {
						const int n0 = n[0] + n[1] + n[2];
						for (m[0] = 0; m[0] <= n[0] / 2; m[0]++) {
							for (m[1] = 0; m[1] <= n[1] / 2; m[1]++) {
								for (m[2] = 0; m[2] <= n[2] / 2; m[2]++) {
									const int m0 = m[0] + m[1] + m[2];
									T num = T(vfactorial(n));
									T den = T((1 << m0) * vfactorial(m) * vfactorial(n - (m) * 2));
									const T fnm = num / den;
									for (k[0] = 0; k[0] <= m0; k[0]++) {
										for (k[1] = 0; k[1] <= m0 - k[0]; k[1]++) {
											k[2] = m0 - k[0] - k[1];
											const auto p = n - (m) * 2 + (k) * 2;
											num = factorial(m0);
											den = vfactorial(k);
											const T number = fnm * num / den;
											Dreal(n) += number * D0(p) * d[n0 - m0] * rinv2pow[m0];
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}

	const auto foursz = econst.nfour();
	for (int i = 0; i < foursz; i++) {
		const auto &h = econst.four_index(i);
		const auto& D0 = econst.four_expansion(i);
		const T hdotx = FMA(h[0], X[0], FMA(h[1], X[1], h[2] * X[2]));                           // 5
		T co;
		T so;
		sincos(T(2.0 * M_PI) * hdotx, &so, &co);
		T soco[2] = { co, so };
		array<int, NDIM> k;
		for (k[0] = 0; k[0] < LORDER; k[0]++) {
			for (k[1] = 0; k[1] < LORDER - k[0]; k[1]++) {
				const int zmax = (k[0] == 0 && k[1] == 0) ? intmin(3, LORDER) : intmin(LORDER - k[0] - k[1], 2);
				for (k[2] = 0; k[2] < zmax; k[2]++) {
					const int k0 = k[0] + k[1] + k[2];
					Dfour(k) += soco[k0 % 2] * D0(k);
				}
			}
		}
	}
	D = Dreal.detraceD() + Dfour;
	expansion<T> D1;
	green_direct(D1, X);
	D(0, 0, 0) = T(M_PI / 4.0) + D(0, 0, 0);                          // 1
	for (int i = 0; i < LP; i++) {                     // 70
		D[i] -= D1[i];
		D[i] *= zero_mask;
	}
	D[0] += 2.837291e+00 * (T(1) - zero_mask);
	if ( LORDER > 2) {
		D[3] += -4.0 / 3.0 * M_PI * (T(1) - zero_mask);
		D[5] += -4.0 / 3.0 * M_PI * (T(1) - zero_mask);
		D[LP - 1] += -4.0 / 3.0 * M_PI * (T(1) - zero_mask);
	}

	return 0;
}
