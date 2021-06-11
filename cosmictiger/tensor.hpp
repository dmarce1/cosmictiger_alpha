#pragma once

#include <cosmictiger/array.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/global.hpp>

CUDA_EXPORT
inline
int factorial(int n) {
	assert(n >= 0);
	if (n == 0) {
		return 1;
	} else {
		return n * factorial(n - 1);
	}
}

CUDA_EXPORT
inline int intmin(int a, int b) {
	return a < b ? a : b;
}

CUDA_EXPORT
inline
int dfactorial(int n) {
	assert(n >= -1);
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
class tensor_trless_sym: public array<T, P * P + 1> {

private:

public:

	static constexpr int N = P * P + 1;

	CUDA_EXPORT
	inline tensor_trless_sym& operator=(const T& other) {
		for (int i = 0; i < N; i++) {
			(*this)[i] = other;
		}
		return *this;
	}

	CUDA_EXPORT
	inline tensor_trless_sym operator+(const tensor_trless_sym& other) const {
		tensor_trless_sym<T, P> result;
		for (int i = 0; i < N; i++) {
			result[i] = other[i] + (*this)[i];
		}
		return result;
	}

	CUDA_EXPORT
	inline tensor_trless_sym operator-(const tensor_trless_sym& other) const {
		tensor_trless_sym<T, P> result;
		for (int i = 0; i < N; i++) {
			result[i] = other[i] - (*this)[i];
		}
		return result;
	}

	CUDA_EXPORT
	inline T& operator()(int l, int m, int n) {
		l += m;
		assert(l >= 0);
		assert(m >= 0);
		assert(n >= 0);
		assert(l < P);
		assert(m <= l);
		assert(n <= 1 || (n == 2 && l == 0 && m == 0));
		return (*this)[l * (l + 1) / 2 + m + (P * (P + 1) / 2) * (n == 1) + (N - 1) * (n == 2)];
	}

	CUDA_EXPORT
	inline T operator()(int l, int m, int n) const {
		if (n > 1) {
			if (l == 0 && m == 0 && n == 2) {
				return (*this)[N - 1];
			} else {
				return -((*this)(l + 2, m, n - 2) + (*this)(l, m + 2, n - 2));
			}
		} else {
			l += m;
			assert(l >= 0);
			assert(m >= 0);
			assert(n >= 0);
			assert(l < P);
			assert(m <= l);
			assert(n <= 1);
			return (*this)[l * (l + 1) / 2 + m + (P * (P + 1) / 2) * n];
		}
	}

	CUDA_EXPORT
	T operator()(const array<int, NDIM>& n) const {
		return (*this)(n[0], n[1], n[2]);
	}

	CUDA_EXPORT
	T& operator()(const array<int, NDIM>& n) {
		return (*this)(n[0], n[1], n[2]);
	}

};

CUDA_EXPORT
inline vector<int> indices_begin(int P) {
	vector<int> v;
	v.reserve(P);
	for (int i = 0; i < P; i++) {
		v.push_back(0);
	}
	return v;
}

CUDA_EXPORT
inline bool indices_inc(vector<int>& i) {
	if (i.size() == 0) {
		return false;
	}
	int j = 0;
	while (i[j] == NDIM - 1) {
		i[j] = 0;
		j++;
		if (j == i.size()) {
			i[0] = -1;
			return false;
		}
	}
	i[j]++;
	return true;
}

CUDA_EXPORT
inline vector<int> indices_end(int P) {
	vector<int> v;
	v.reserve(P);
	v.push_back(-1);
	for (int i = 1; i < P; i++) {
		v.push_back(0);
	}
	return v;

}

CUDA_EXPORT
inline array<int, NDIM> indices_to_sym(const vector<int>& indices) {
	array<int, NDIM> j;
	j[0] = j[1] = j[2] = 0;
	for (int i = 0; i < indices.size(); i++) {
		j[indices[i]]++;
	}
	return j;
}

CUDA_EXPORT
inline vector<int> sym_to_indices(const array<int, NDIM>& i) {
	vector<int> indices;
	indices.reserve(i[0] + i[1] + i[2]);
	for (int dim = 0; dim < NDIM; dim++) {
		for (int j = 0; j < i[dim]; j++) {
			indices.push_back(dim);
		}
	}
	return indices;
}

template<class T, int P>
class tensor_sym: public array<T, (P * (P + 1) * (P + 2)) / 6> {

private:

public:

	static constexpr int N = (P * (P + 1) * (P + 2)) / 6;

	CUDA_EXPORT
	inline tensor_sym& operator=(const T& other) {
		for (int i = 0; i < N; i++) {
			(*this)[i] = other;
		}
		return *this;
	}

	CUDA_EXPORT
	inline tensor_sym operator+(const tensor_sym& other) const {
		tensor_sym<T, P> result;
		for (int i = 0; i < N; i++) {
			result[i] = other[i] + (*this)[i];
		}
		return result;
	}

	CUDA_EXPORT
	inline T operator()(int l, int m, int n) const {
		m += n;
		l += m;
		assert(l >= 0);
		assert(m >= 0);
		assert(n >= 0);
		assert(l < P);
		assert(m <= l);
		assert(n <= m);
		return (*this)[l * (l + 1) * (l + 2) / 6 + m * (m + 1) / 2 + n];
	}

	CUDA_EXPORT
	inline T& operator()(int l, int m, int n) {
		m += n;
		l += m;
		assert(l >= 0);
		assert(m >= 0);
		assert(n >= 0);
		assert(l < P);
		assert(m <= l);
		assert(n <= m);
		return (*this)[l * (l + 1) * (l + 2) / 6 + m * (m + 1) / 2 + n];
	}

	CUDA_EXPORT
	T operator()(const array<int, NDIM>& n) const {
		return (*this)(n[0], n[1], n[2]);
	}

	CUDA_EXPORT
	T& operator()(const array<int, NDIM>& n) {
		return (*this)(n[0], n[1], n[2]);
	}

	CUDA_EXPORT
	tensor_trless_sym<T, P> detraceF() const {
		tensor_trless_sym<T, P> A;
		const tensor_sym<T, P>& B = *this;
		array<int, NDIM> m;
		array<int, NDIM> k;
		array<int, NDIM> n;
		for (n[0] = 0; n[0] < P; n[0]++) {
			for (n[1] = 0; n[1] < P - n[0]; n[1]++) {
				const int nzmax = (n[0] == 0 && n[1] == 0) ? intmin(3, P) : intmin(P - n[0] - n[1], 2);
				for (n[2] = 0; n[2] < nzmax; n[2]++) {
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
										const auto p = n - (m) * 2 + (k) * 2;
										num = factorial(m0);
										den = vfactorial(k);
										const T number = fnm * num / den;
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

	CUDA_EXPORT
	inline tensor_trless_sym<T, P> detraceD() const {
		tensor_trless_sym<T, P> A;
		const tensor_sym<T, P>& B = *this;
		array<int, NDIM> m;
		array<int, NDIM> k;
		array<int, NDIM> n;
		for (n[0] = 0; n[0] < P; n[0]++) {
			for (n[1] = 0; n[1] < P - n[0]; n[1]++) {
				const int nzmax = (n[0] == 0 && n[1] == 0) ? intmin(3, P) : intmin(P - n[0] - n[1], 2);
				for (n[2] = 0; n[2] < nzmax; n[2]++) {
					A(n) = T(0);
					const int n0 = n[0] + n[1] + n[2];
					for (m[0] = 0; m[0] <= n[0] / 2; m[0]++) {
						for (m[1] = 0; m[1] <= n[1] / 2; m[1]++) {
							for (m[2] = 0; m[2] <= n[2] / 2; m[2]++) {
								const int m0 = m[0] + m[1] + m[2];
								T num = T(n1pow(m0) * dfactorial(2 * n0 - 2 * m0 - 1) * vfactorial(n));
								T den = T((1 << m0) * vfactorial(m) * vfactorial(n - (m) * 2));
								const T fnm = num / den;
								if ((n0 == 2 && (n[0] == 2 || n[1] == 2 || n[2] == 2)) && m0 == 1) {
									continue;
								}
								for (k[0] = 0; k[0] <= m0; k[0]++) {
									for (k[1] = 0; k[1] <= m0 - k[0]; k[1]++) {
										k[2] = m0 - k[0] - k[1];
										const auto p = n - (m) * 2 + (k) * 2;
										num = factorial(m0);
										den = vfactorial(k);
										const T number = fnm * num / den;
										A(n) += number * B(p);
									}
								}
							}
						}
					}
					A(n) /= T(dfactorial(2 * n0 - 1));
				}
			}
		}
		return A;
	}
};

template<class T, int P>
CUDA_EXPORT
inline tensor_trless_sym<T, P> rotate(const tensor_trless_sym<T, P> & B, T theta, bool pitch) {
	tensor_trless_sym<T, P> A;
	array<int, NDIM> n;
	array<array<T, NDIM>, NDIM> R;
	const T cos0 = cos(theta);
	const T sin0 = sin(theta);
	if (!pitch) {
		R[0][0] = cos0;
		R[1][0] = -sin0;
		R[2][0] = T(0);
		R[0][1] = sin0;
		R[1][1] = cos0;
		R[2][1] = T(0);
		R[0][2] = T(0);
		R[1][2] = T(0);
		R[2][2] = T(1);
	} else {
		R[0][0] = cos0;
		R[1][0] = T(0);
		R[2][0] = -sin0;
		R[0][1] = T(0);
		R[1][1] = T(1);
		R[2][1] = T(0);
		R[0][2] = sin0;
		R[1][2] = T(0);
		R[2][2] = cos0;
	}
	for (n[0] = 0; n[0] < P; n[0]++) {
		for (n[1] = 0; n[1] < P - n[0]; n[1]++) {
			const int nzmax = (n[0] == 0 && n[1] == 0) ? intmin(3, P) : intmin(P - n[0] - n[1], 2);
			for (n[2] = 0; n[2] < nzmax; n[2]++) {
				A(n) = T(0);
				const int n0 = n[0] + n[1] + n[2];
				const auto even_indices = sym_to_indices(n);
				if (n0 == 0) {
					A(n) = B(n);
				} else {
		//			printf("\n");
					for (auto odd_indices = indices_begin(n0); odd_indices[0] != -1; indices_inc(odd_indices)) {
						T factor = T(1);
						for (int l = 0; l < n0; l++) {
							factor *= R[odd_indices[l]][even_indices[l]];
		//					printf("%i %i %i %i %i\n", n[0], n[1], n[2], odd_indices[l], even_indices[l]);
						}
						A(n) += factor * B(indices_to_sym(odd_indices));
					}
				}
			}
		}
	}
	return A;
}

template<class T, int P>
CUDA_EXPORT
tensor_sym<T, P> vector_to_sym_tensor(const array<T, NDIM>& vec) {
	tensor_sym<T, P> X;
	array<int, NDIM> n;
	T x = T(1);
	for (n[0] = 0; n[0] < P; n[0]++) {
		T y = T(1);
		for (n[1] = 0; n[1] < P - n[0]; n[1]++) {
			T z = T(1);
			for (n[2] = 0; n[2] < P - n[0] - n[1]; n[2]++) {
				X(n) = x * y * z;
				z *= vec[2];
			}
			y *= vec[1];
		}
		x *= vec[0];
	}
	return X;
}

template<class T, int P, int Q = P>
CUDA_EXPORT
tensor_trless_sym<T, P> monopole_translate(const array<T, NDIM>& x) {
	return vector_to_sym_tensor<T, P>(-x).detraceD();
}

template<class T, int P>
CUDA_EXPORT
tensor_trless_sym<T, P> multipole_translate(const tensor_trless_sym<T, P>& M1, const array<T, NDIM>& x) {
	tensor_sym<T, P> M2;
	array<int, NDIM> k;
	array<int, NDIM> n;
	const auto delta_x = vector_to_sym_tensor<T, P>(-x);
	for (n[0] = 0; n[0] < P; n[0]++) {
		for (n[1] = 0; n[1] < P - n[0]; n[1]++) {
			for (n[2] = 0; n[2] < P - n[0] - n[1]; n[2]++) {
				const int n0 = n[0] + n[1] + n[2];
				M2(n) = M1(n);
			}
		}
	}
	for (int n0 = P - 1; n0 >= 0; n0--) {
		for (n[0] = 0; n[0] <= n0; n[0]++) {
			for (n[1] = 0; n[1] <= n0 - n[0]; n[1]++) {
				n[2] = n0 - n[0] - n[1];
				for (k[0] = 0; k[0] <= intmin(n0, n[0]); k[0]++) {
					for (k[1] = 0; k[1] <= intmin(n0 - k[0], n[1]); k[1]++) {
						for (k[2] = 0; k[2] <= intmin(n0 - k[0] - k[1], n[2]); k[2]++) {
							const auto factor = T(vfactorial(n)) / T(vfactorial(k) * vfactorial(n - k));
							if (n != k) {
								M2(n) += factor * delta_x(n - k) * M2(k);
							}
						}
					}
				}
			}
		}
	}
	return M2.detraceD();
}

template<class T, int P>
CUDA_EXPORT
tensor_trless_sym<T, P> direct_greens_function(const array<T, NDIM> x) {
	auto D = vector_to_sym_tensor<T, P>(x).detraceF();
	array<int, NDIM> k;
	array<T, P> rinv_pow;
	const auto r2 = sqr(x[0], x[1], x[2]);
	const auto r = sqrt(r2);
	const auto rinv = (r > T(0)) / max(r, 1e-20);
	const auto rinv2 = rinv * rinv;
	rinv_pow[0] = -rinv;
	for (int i = 1; i < P; i++) {
		rinv_pow[i] = -rinv2 * rinv_pow[i - 1];
	}
	for (k[0] = 0; k[0] < P; k[0]++) {
		for (k[1] = 0; k[1] < P - k[0]; k[1]++) {
			const int zmax = (k[0] == 0 && k[1] == 0) ? intmin(3, P) : intmin(P - k[0] - k[1], 2);
			for (k[2] = 0; k[2] < zmax; k[2]++) {
				const int k0 = k[0] + k[1] + k[2];
				D(k) *= rinv_pow[k0];
			}
		}
	}
	return D;
}

#include <cosmictiger/simd.hpp>

inline float first(simd_float a) {
	return a[0];
}

CUDA_EXPORT
inline float first(float a) {
	return a;
}

template<class T, int P, int Q>
CUDA_EXPORT
tensor_trless_sym<T, P> interaction(tensor_trless_sym<T, Q> M0, const tensor_trless_sym<T, Q + 1>& D) {
	tensor_trless_sym<T, P> L;
	array<int, NDIM> n;
	array<int, NDIM> m;
	M0 = rotate(M0, T(1.0), true);
	const auto M = rotate(M0, T(-1.0), true);
//	const auto M = M0;
	for (n[0] = 0; n[0] < P; n[0]++) {
		for (n[1] = 0; n[1] < P - n[0]; n[1]++) {
			const int nzmax = (n[0] == 0 && n[1] == 0) ? intmin(3, P) : intmin(P - n[0] - n[1], 2);
			for (n[2] = 0; n[2] < nzmax; n[2]++) {
				L(n) = T(0);
				const int n0 = n[0] + n[1] + n[2];
				const int q0 = intmin(Q + 1 - n0, Q);
				for (m[0] = 0; m[0] < q0; m[0]++) {
					for (m[1] = 0; m[1] < q0 - m[0]; m[1]++) {
						for (m[2] = 0; m[2] < q0 - m[0] - m[1]; m[2]++) {
							L(n) += M(m) * D(n + m) / T(vfactorial(m));
						}
					}
				}
			}
		}
	}
	return L;
}

template<class T, int P, int Q = P>
CUDA_EXPORT
tensor_trless_sym<T, P> expansion_translate(const tensor_trless_sym<T, Q> L1, const array<T, NDIM>& x) {
	tensor_trless_sym<T, P> L2;
	array<int, NDIM> k;
	array<int, NDIM> n;
	const auto delta_x = vector_to_sym_tensor<T, Q>(x);
	for (n[0] = 0; n[0] < P; n[0]++) {
		for (n[1] = 0; n[1] < P - n[0]; n[1]++) {
			const int nzmax = (n[0] == 0 && n[1] == 0) ? intmin(3, P) : intmin(P - n[0] - n[1], 2);
			for (n[2] = 0; n[2] < nzmax; n[2]++) {
				L2(n) = L1(n);
			}
		}
	}
	for (int n0 = 0; n0 < P; n0++) {
		for (n[0] = 0; n[0] <= n0; n[0]++) {
			for (n[1] = 0; n[1] <= n0 - n[0]; n[1]++) {
				n[2] = n0 - n[1] - n[0];
				if (n[2] <= 1 || (n[0] == 0 && n[1] == 0 && n[2] == 2)) {
					const int n0 = n[0] + n[1] + n[2];
					for (k[0] = 0; k[0] < Q - n0; k[0]++) {
						for (k[1] = 0; k[1] < Q - n0 - k[0]; k[1]++) {
							for (k[2] = 0; k[2] < Q - n0 - k[0] - k[1]; k[2]++) {
								const auto factor = T(1) / T(vfactorial(k));
								const auto p = n + k;
								const int p0 = p[0] + p[1] + p[2];
								if (n != p) {
									L2(n) += factor * delta_x(k) * L1(p);
								}
							}
						}
					}
				}
			}
		}
	}
	return L2;
}
