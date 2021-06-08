#pragma once

#include <cosmictiger/array.hpp>

CUDA_EXPORT
inline
int factorial(int n) {
	assert(n>=0);
	if (n == 0) {
		return 1;
	} else {
		return n * factorial(n - 1);
	}
}

CUDA_EXPORT
inline
int dfactorial(int n) {
	assert(n>=-1);
	if (n >= -1 && n <= 1) {
		return 1;
	} else {
		return n * dfactorial(n - 2);
	}
}

template<int N>
CUDA_EXPORT inline
int vfactorial(const array<int, N>& n) {
	return factorial(n[0]) * factorial(n[1]) * factorial(n[2]);
}

CUDA_EXPORT
inline
int n1pow(int n) {
	return (n & 1) ? -1 : 1;
}

template<class T, int P>
class tensor_sym: public array<T, P * (P + 1) * (P + 2) / 6> {

private:

public:

	static constexpr int N = P * (P + 1) * (P + 1) / 6;

	CUDA_EXPORT
	inline T operator()(int l, int m, int n) const {
		m += n;
		l += m;
		assert(l>=0);
		assert(m>=0);
		assert(n>=0);
		assert(l < P);
		assert(m <= l);
		assert(n <= m);
		return (*this)[l * (l + 1) * (l + 2) / 6 + m * (m + 1) / 2 + n];
	}

	CUDA_EXPORT
	inline T& operator()(int l, int m, int n) {
		m += n;
		l += m;
		assert(l>=0);
		assert(m>=0);
		assert(n>=0);
		assert(l < P);
		assert(m <= l);
		assert(n <= m);
		return (*this)[l * (l + 1) * (l + 2) / 6 + m * (m + 1) / 2 + n];
	}

	CUDA_EXPORT
	T operator()(const array<int, N>& n) const {
		return (*this)(n[0], n[1], n[2]);
	}

	CUDA_EXPORT
	T& operator()(const array<int, N>& n) {
		return (*this)(n[0], n[1], n[2]);
	}

	tensor_sym<T, P> detraceF() const {
		tensor_sym<T, P> A;
		const tensor_sym<T, P>& B = *this;
		array<int, N> n;
		array<int, N> m;
		array<int, N> k;
		for (n[0] = 0; n[0] < P; n[0]++) {
			for (n[1] = 0; n[1] < P - n[0]; n[1]++) {
				for (n[2] = 0; n[2] < P - n[0] - n[1]; n[2]++) {
					A(n) = T(0);
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
										const int k0 = m0;
										const auto p = n - (m) * 2 + (k) * 2;
										num = factorial(m0) ;
										den = vfactorial(k);
										const T number = fnm * num / den;
								//		printf( "%i %i %i %i %i %i %e %e\n", n[0], n[1], n[2], p[0], p[1], p[2], (float) dfactorial(2 * n0 - 2 * m0 - 1) , (float) (2 * n0 - 2 * m0 - 1) );
										A(n) += number * B(p);
									}
								}
							}
						}
					}
				}
			}
		}
		return A;
	}
};
