/*
 * spherical_harmonic.hpp
 *
 *  Created on: Jun 2, 2021
 *      Author: dmarce1
 */

#ifndef SPHERICAL_HARMONIC_HPP_
#define SPHERICAL_HARMONIC_HPP_

#include <cosmictiger/array.hpp>
#include <cosmictiger/cuda.hpp>
#include <cosmictiger/math.hpp>
#include <math.h>

using std::max;

using std::min;

#define PMAX 10
#define FACT_MAX 34

struct sphericalYconstants;

template<class T, int P>
struct sphericalY: public array<complex<T>, P * (P + 1) / 2> {
	CUDA_EXPORT
	inline complex<T> operator()(int l, int m) const {
		assert(l >= 0);
		assert(l < P);
		assert(m >= -l);
		assert(m <= l);
		return m < 0 ? (*this)[((l * (l + 1)) >> 1) - m].conj() : (*this)[((l * (l + 1)) >> 1) + m];
	}
	CUDA_EXPORT
	inline complex<T>& operator()(int l, int m) {
		assert(l >= 0);
		assert(l < P);
		assert(m >= 0);
		assert(m <= l);
		return (*this)[((l * (l + 1)) >> 1) + m];
	}
	CUDA_EXPORT
	inline complex<T> operator()(int l) const {
		assert(l >= 0);
		assert(l < P);
		return (*this)(l, 0);
	}
	CUDA_EXPORT
	inline complex<T>& operator()(int l) {
		assert(l >= 0);
		assert(l < P);
		return (*this)(l, 0);
	}
	CUDA_EXPORT
	inline sphericalY<T, P> operator+(const sphericalY<T, P>& other) const {
		sphericalY<T, P> result;
		for (int i = 0; i < P * (P + 1) / 2; i++) {
			result[i] = (*this)[i] + other[i];
		}
		return result;
	}
	CUDA_EXPORT
	inline sphericalY<T, P> operator-(const sphericalY<T, P>& other) const {
		sphericalY<T, P> result;
		for (int i = 0; i < P * (P + 1) / 2; i++) {
			result[i] = (*this)[i] - other[i];
		}
		return result;
	}
	CUDA_EXPORT
	inline sphericalY<T, P> operator*(const T& other) const {
		sphericalY<T, P> result;
		for (int i = 0; i < P * (P + 1) / 2; i++) {
			result[i] = (*this)[i] * other;
		}
		return result;
	}
	CUDA_EXPORT
	inline sphericalY<T, P> operator/(const T& other) const {
		sphericalY<T, P> result;
		for (int i = 0; i < P * (P + 1) / 2; i++) {
			result[i] = (*this)[i] / other;
		}
		return result;
	}
	CUDA_EXPORT
	inline sphericalY<T, P>& operator=(const sphericalY<T, P>& other) {
		for (int i = 0; i < P * (P + 1) / 2; i++) {
			(*this)[i] = other[i];
		}
		return *this;
	}
	CUDA_EXPORT
	inline sphericalY<T, P>& operator+=(const sphericalY<T, P>& other) {
		(*this) = (*this) + other;
		return *this;
	}
	CUDA_EXPORT
	inline sphericalY<T, P>& operator-=(const sphericalY<T, P>& other) {
		(*this) = (*this) - other;
		return *this;
	}
	CUDA_EXPORT
	inline sphericalY<T, P>& operator*=(const T& other) {
		(*this) = (*this) * other;
		return *this;
	}
	CUDA_EXPORT
	inline sphericalY<T, P>& operator/=(const T& other) {
		(*this) = (*this) / other;
		return *this;
	}
	CUDA_EXPORT
	inline sphericalY<T, P>& operator=(const T& other) {
		for (int i = 0; i < P * (P + 1) / 2; i++) {
			(*this)[i] = other;
		}
		return *this;
	}
};

struct sphericalYconstants {
	array<float, FACT_MAX> factorial;
	array<float, PMAX> lp1inv;
	array<float, PMAX * (PMAX + 1) / 2> Ynorm;
	array<float, PMAX * (PMAX + 1) / 2> bigA;
	array<float, PMAX * (PMAX + 1) / 2> bigAinv;
	array<cmplx, 4> ipow;
};

#ifndef SPHERICAL_HARMONICS_CPP
#ifdef __CUDA_ARCH__
extern __constant__ sphericalYconstants gpu_spherical_constants;
static __device__ auto& spherical_constants = gpu_spherical_constants;
#else
extern sphericalYconstants cpu_spherical_constants;
static auto& spherical_constants = cpu_spherical_constants;
#endif
#endif

void spherical_harmonics_init_gpu(const sphericalYconstants& constants);
void spherical_harmonics_init();

CUDA_EXPORT inline float factorial(int n) {
	return spherical_constants.factorial[n];
}

CUDA_EXPORT inline float lp1inv(int n) {
	return spherical_constants.lp1inv[n];
}

CUDA_EXPORT inline float bigA(int n, int m) {
	return spherical_constants.bigA[((n * (n + 1)) >> 1) + abs(m)];
}

CUDA_EXPORT inline float bigAinv(int n, int m) {
	return spherical_constants.bigAinv[((n * (n + 1)) >> 1) + abs(m)];
}

CUDA_EXPORT inline float Ynorm(int n, int m) {
	return spherical_constants.Ynorm[((n * (n + 1)) >> 1) + abs(m)];
}

CUDA_EXPORT inline cmplx ipow(int n) {
	return spherical_constants.ipow[(n + 100) % 4];
}

CUDA_EXPORT inline float n1pow(int n) {
	return (n & 1) ? -1.f : 1.f;
}

template<class T, int P>
CUDA_EXPORT inline void spherical_harmonic_helper(sphericalY<T, P>& Y, T x, T y, T z, T& r, T& rinv) {
	const T R2 = fmaf(x, x, sqr(y));
	const T r2 = R2 + sqr(z);
	r = sqrt(r2);
	const T R = sqrt(R2);
	const T eps(1.0e-10);
	rinv = 1.0f / max(r, eps);
	const T Rinv = 1.0f / max(R, eps);
	T cos0 = min(max(T(-1), z * rinv), T(+1));
	T sin0 = sqrt(T(1) - cos0 * cos0);
	T csc0 = T(1) / max(sin0, eps);
	Y(0) = complex<T>(1.0f, 0.0);
	if (P > 1) {
		Y(1) = complex<T>(cos0, 0.0);
	}
	for (int l = 1; l < P - 1; l++) {
		Y(l + 1) = (T(2 * l + 1) * cos0 * Y(l) - T(l) * Y(l - 1)) * lp1inv(l);
	}
	for (int l = 1; l < P; l++) {
		for (int m = 0; m < l; m++) {
			Y(l, m + 1) = (T(l - m) * cos0 * Y(l, m) - T(l + m) * Y(l - 1, m)) * csc0;
		}
	}
	for (int l = 1; l < P; l++) {
		complex<T> Rpow = complex<T>(x, y) * Rinv;
		for (int m = 1; m <= l; m++) {
			Y(l, m) *= Rpow;
			Rpow *= complex<T>(x, y) * Rinv;
		}
	}
	for (int l = 0; l < P; l++) {
		for (int m = 0; m <= l; m++) {
			Y(l, m) *= Ynorm(l, m);
		}
	}
}

template<class T, int P>
CUDA_EXPORT inline void spherical_harmonic(sphericalY<T, P>& Y, T x, T y, T z) {
	T r, rinv;
	spherical_harmonic_helper(Y, x, y, z, r, rinv);

}

template<class T, int P>
CUDA_EXPORT inline void regular_harmonic(sphericalY<T, P>& Y, T x, T y, T z) {
	T r, rinv;
	spherical_harmonic_helper(Y, x, y, z, r, rinv);
	T rpow = T(1);
	for (int l = 0; l < P; l++) {
		for (int m = 0; m <= l; m++) {
			Y(l, m) *= rpow;
		}
		rpow *= r;
	}

}
template<class T, int P>
CUDA_EXPORT inline void irregular_harmonic(sphericalY<T, P>& Y, T x, T y, T z) {
	T r, rinv;
	spherical_harmonic_helper(Y, x, y, z, r, rinv);
	T rpow = T(rinv);
	for (int l = 0; l < P; l++) {
		for (int m = 0; m <= l; m++) {
			Y(l, m) *= rpow;
		}
		rpow *= rinv;
	}
}

CUDA_EXPORT inline constexpr int cmax(int a, int b) {
	return a < b ? b : a;
}

CUDA_EXPORT inline constexpr int cmin(int a, int b) {
	return a > b ? b : a;
}

template<class T, int P, int Q>
CUDA_EXPORT inline void translate_multipole(sphericalY<T, P>& M, const sphericalY<T, Q> O, T x, T y, T z) {
	sphericalY<T, cmax(P, Q)> R0;
	regular_harmonic(R0, x, y, z);
	const auto R = R0;
	for (int j = 0; j < P; j++) {
		for (int k = 0; k <= j; k++) {
			for (int n = 0; n <= j; n++) {
				if (j - n >= Q) {
					continue;
				}
				for (int m = -n; m <= n; m++) {
					if (abs(k - m) > j - n) {
						continue;
					}
					M(j, k) += O(j - n, k - m) * ipow(abs(k) - abs(m) - abs(k - m)) * bigA(n, m) * bigA(j - n, k - m)
							* bigAinv(j, k) * R(n, -m);
				}
			}
		}
	}

}

template<class T, int P, int Q>
CUDA_EXPORT inline void translate_expansion(sphericalY<T, P>& L, const sphericalY<T, Q> O, T x, T y, T z) {
	sphericalY<T, cmax(P, Q)> R0;
	spherical_harmonic(R0, x, y, z);
	const auto R = R0;
	for (int j = 0; j < P; j++) {
		for (int k = 0; k <= j; k++) {
			for (int n = j; n < Q; n++) {
				for (int m = -n; m <= n; m++) {
					if (abs(k - m) > n - j) {
						continue;
					}
					L(j, k) += O(n, m) * ipow(abs(m) - abs(k) - abs(k - m)) * bigA(n - j, m - k) * bigA(j, k) * bigAinv(n, m)
							* R(n - j, m - k) * n1pow(n + j);
				}
			}
		}
	}
}

template<class T, int P, int Q, int R>
CUDA_EXPORT inline void sph_multipole_interaction(sphericalY<T, P>& L, const sphericalY<T, Q>& M,
		const sphericalY<T, R>& I) {
	for (int j = 0; j < P; j++) {
		for (int k = 0; k <= j; k++) {
			for (int n = 0; n < Q; n++) {
				if (n + j >= R) {
					continue;
				}
				for (int m = -n; m <= n; m++) {
					if (abs(m - k) > j + n) {
						continue;
					}
					L(j, k) += M(n, m) * ipow(abs(k - m) - abs(k) - abs(m)) * bigA(n, m) * bigA(j, k) * I(j + n, m - k)
							* n1pow(n) * bigAinv(j + n, m - k);
				}
			}
		}
	}
}

#endif /* SPHERICAL_HARMONIC_HPP_ */
