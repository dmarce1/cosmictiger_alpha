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
#include <cosmictiger/simd.hpp>
#include <cosmictiger/math.hpp>
#include <math.h>

using std::max;

using std::min;

#define PMAX 10
#define FACT_MAX 34

struct sphericalYconstants;

template<class T, int P>
struct sphericalY: public array<complex<T>, P * (P + 1) / 2> {
	/*	CUDA_EXPORT
	 sphericalY() {
	 for (int i = 0; i < P * (P + 1) / 2; i++) {
	 (*this)[i] = complex<T>(T(1e99), T(1e99));
	 }
	 }*/
	CUDA_EXPORT
	inline complex<T> operator()(int l, int m) const {
		assert(l >= 0);
		assert(l < P);
		assert(m >= -l);
		assert(m <= l);
		if (m < 0) {
			return (*this)[((l * (l + 1)) >> 1) - m].conj();
		} else {
			return (*this)[((l * (l + 1)) >> 1) + m];
		}
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
	template<int Q>
	CUDA_EXPORT
	inline sphericalY<T, P>& operator=(const sphericalY<T, Q>& other) {
		for (int l = 0; l < min(P, Q); l++) {
			for (int m = 0; m <= l; m++) {
				(*this)(l, m) = other(l, m);
			}
		}
		for (int l = min(P, Q); l < P; l++) {
			for (int m = 0; m <= l; m++) {
				(*this)(l, m) = complex<T>(T(0), T(0));
			}
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
extern __managed__ sphericalYconstants gpu_spherical_constants;
#define spherical_constants  gpu_spherical_constants
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

#include <cosmictiger/simd.hpp>

template<class TYPE, int P>
CUDA_EXPORT inline void spherical_harmonic_helper(sphericalY<TYPE, P>& Y, TYPE x, TYPE y, TYPE z, TYPE& r, TYPE& rinv) {
	const TYPE R2 = fmaf(x, x, sqr(y));
	const TYPE r2 = fmaf(z, z, R2);
	r = sqrt(r2);
	const TYPE R = sqrt(R2);
	const TYPE eps(1.0e-10);
	const TYPE huge(1.0e+9);
	rinv = 1.0f / max(r, eps);
	const TYPE Rinv = 1.0f / max(R, eps);
	TYPE cos0 = min(max(TYPE(-1), z * rinv), TYPE(+1));
	TYPE sin0 = sqrt(TYPE(1) - cos0 * cos0);
	TYPE csc0 = TYPE(1) / max(sin0, eps);
	csc0 *= (csc0 < huge);
	Y(0) = complex<TYPE>(1.0f, 0.0);
	if (P > 1) {
		Y(1) = complex<TYPE>(cos0, 0.0);
	}
	for (int l = 1; l < P - 1; l++) {
		Y(l + 1) = (TYPE(2 * l + 1) * cos0 * Y(l) - TYPE(l) * Y(l - 1)) * lp1inv(l);
	}
	for (int l = 1; l < P; l++) {
		for (int m = 0; m < l; m++) {
			Y(l, m + 1) = (TYPE(l - m) * cos0 * Y(l, m) - TYPE(l + m) * Y(l - 1, m)) * csc0;
		}
	}
	for (int l = 1; l < P; l++) {
		complex<TYPE> Rpow = complex<TYPE>(x, y) * Rinv;
		for (int m = 1; m <= l; m++) {
			Y(l, m) *= Rpow;
			Rpow *= complex<TYPE>(x, y) * Rinv;
		}
	}
	for (int l = 0; l < P; l++) {
		for (int m = 0; m <= l; m++) {
			Y(l, m) *= Ynorm(l, m);
		}
	}
//	if( P > 1 ) {
//		printf( "%e\n", first(Y(1,1).real())/first(x/r));
//	}
}

template<class T, int P>
CUDA_EXPORT inline void spherical_harmonic(sphericalY<T, P>& Y, T x, T y, T z) {
	T r, rinv;
	spherical_harmonic_helper(Y, x, y, z, r, rinv);

}

template<class T, int P>
CUDA_EXPORT inline void regular_harmonic(sphericalY<T, P>& Y, T x, T y, T z) {
	T R2 = x * x + y * y;
	T r2 = x * x + y * y + z * z;
	T r = sqrt(r2);
	T Rinv = T(1) / (max(sqrt(R2), T(1.0e-20)));
	T cos0 = z / r;
	T sin0 = sqrt(T(1) - cos0 * cos0);
	T fact = 1;
	T pn = 1;
	T rm = 1;
	complex<T> ei = complex<T>(x * Rinv, y * Rinv);
	complex<T> eim = 1.0;
	for (int m = 0; m < P; m++) {
		T p = pn;
		Y(m, m) = rm * p * eim;
		T p1 = p;
		p = cos0 * (2 * m + 1) * p1;
		rm *= r;
		T rn = rm;
		for (int n = m + 1; n < P; n++) {
			rn /= -(n + m);
			Y(n, m) = rn * p * eim;
			T p2 = p1;
			p1 = p;
			p = (cos0 * (2 * n + 1) * p1 - (n + m) * p2) / (n - m + 1);
			rn *= r;
		}
		rm /= -(2 * m + 2) * (2 * m + 1);
		pn = -pn * fact * sin0;
		fact += 2;
		eim *= ei;
	}
}

template<class T, int P>
CUDA_EXPORT inline void irregular_harmonic(sphericalY<T, P>& Y, T x, T y, T z) {
	T R2 = x * x + y * y;
	T r2 = x * x + y * y + z * z;
	T r = sqrt(r2);
	T Rinv = T(1) / (max(sqrt(R2), 1.0e-20));
	T fact = 1;
	T pn = 1;
	T invR = -1.0 / r;
	T rm = -invR;
	T cos0 = z / r;
	T sin0 = sqrt(T(1) - cos0 * cos0);
	complex<T> ei = complex<T>(x * Rinv, y * Rinv);
	complex<T> eim = T(1);
	for (int m = 0; m < P; m++) {
		T p = pn;
		Y(m, m) = rm * p * eim;
		T p1 = p;
		p = cos0 * (2 * m + 1) * p1;
		rm *= invR;
		T rhon = rm;
		for (int n = m + 1; n < P; n++) {
			Y(n, m) = rhon * p * eim;
			T p2 = p1;
			p1 = p;
			p = (cos0 * (2 * n + 1) * p1 - (n + m) * p2) / (n - m + 1);
			rhon *= invR * (n - m + 1);
		}
		pn = -pn * fact * sin0;
		fact += 2;
		eim *= ei;
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
			M(j, k) = T(0);
			for (int n = 0; n <= j; n++) {
				for (int n = 0; n <= j; n++) {
					if (j - n >= Q) {
						continue;
					}
					for (int m = std::max(-n, -j + k + n); m <= std::min(k - 1, n); m++) {
						int jnkms = (j - n) * (j - n + 1) / 2 + k - m;
						int nm = n * n + n - m;
						M(j, k) += O(j - n, k - m) * R(n, -m) * ipow(m) * n1pow(n);
					}
					for (int m = k; m <= std::min(n, j + k - n); m++) {
						int jnkms = (j - n) * (j - n + 1) / 2 - k + m;
						int nm = n * n + n - m;
						M(j, k) += O(j - n, k - m) * R(n, -m) * n1pow(k + n + m);
					}
				}
			}
		}
	}
}

template<class T, int P, int Q>
CUDA_EXPORT inline void translate_expansion(sphericalY<T, P>& L, const sphericalY<T, Q> O, T x, T y, T z) {
	sphericalY<T, cmax(P, Q)> R0;
	regular_harmonic(R0, x, y, z);
	const auto R = R0;
	for (int j = 0; j < P; j++) {
		for (int k = 0; k <= j; k++) {
			L(j, k) = 0;
			for (int n = j; n < Q; n++) {
				for (int m = j + k - n; m < 0; m++) {
					L(j, k) += O(n, m) * R(n - j, m - k) * n1pow(k);
				}
				for (int m = 0; m <= n; m++) {
					if (n - j >= abs(m - k)) {
						L(j, k) += O(n, m) * R(n - j, m - k) * n1pow((m - k) * (m < k));
					}
				}
			}
		}
	}
}

template<class T, int P, int Q, int R>
CUDA_EXPORT inline void sph_multipole_interaction(sphericalY<T, P>& L, const sphericalY<T, Q>& M,
		const sphericalY<T, R>& I) {
	for (int j = 0; j < P; j++) {
		T Cnm = n1pow(j);
		for (int k = 0; k <= j; k++) {
			for (int n = 0; n < Q; n++) {
				if (j + n >= R) {
					continue;
				}
				for (int m = -n; m < 0; m++) {
					L(j, k) += M(n, m) * Cnm * I(j + n, m - k);
				}
				for (int m = 0; m <= n; m++) {
					T Cnm2 = Cnm * n1pow((k - m) * (k < m) + m);
					L(j, k) += M(n, m) * Cnm2 * I(j + n, m - k);
				}
			}
		}
	}
}

#endif /* SPHERICAL_HARMONIC_HPP_ */
