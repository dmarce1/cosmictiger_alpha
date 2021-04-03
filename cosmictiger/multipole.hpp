/*
 * multipole_type.hpp
 *
 *  Created on: Jan 30, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_multipole_type_HPP_
#define COSMICTIGER_multipole_type_HPP_

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/array.hpp>
#include <cosmictiger/fixed.hpp>
#include <array>

constexpr int MP = 17;

template<class T>
class multipole_type {
private:
	array<T, MP> data;
public:
	CUDA_EXPORT
	multipole_type();CUDA_EXPORT
	T operator ()() const;CUDA_EXPORT
	T& operator ()();CUDA_EXPORT
	T operator ()(int i, int j) const;CUDA_EXPORT
	T& operator ()(int i, int j);CUDA_EXPORT
	T operator ()(int i, int j, int k) const;CUDA_EXPORT
	T& operator ()(int i, int j, int k);CUDA_EXPORT
	multipole_type<T>& operator =(const multipole_type<T> &other);CUDA_EXPORT
	multipole_type<T>& operator =(T other);
	template<class V>
	CUDA_EXPORT inline multipole_type<T> operator>>(const std::array<V, NDIM> &Y) const;CUDA_EXPORT
	inline multipole_type<T> operator +(const multipole_type<T> &vec) const;
	template<class V>
	CUDA_EXPORT inline multipole_type<T>& operator>>=(const std::array<V, NDIM> &Y);CUDA_EXPORT
	T& operator[](int i) {
		return data[i];
	}
	CUDA_EXPORT
	const T operator[](int i) const {
		return data[i];
	}

	template<class A>
	void serialize(A &&arc, unsigned) {
		for (int i = 0; i < MP; i++) {
			arc & data[i];
		}
	}

};

template<class T>
CUDA_EXPORT inline multipole_type<T>::multipole_type() {
}

template<class T>
CUDA_EXPORT inline T multipole_type<T>::operator ()() const {
	return data[0];
}

template<class T>
CUDA_EXPORT inline T& multipole_type<T>::operator ()() {
	return data[0];
}

template<class T>
CUDA_EXPORT inline T multipole_type<T>::operator ()(int i, int j) const {
	static constexpr size_t map2[3][3] = { { 0, 1, 2 }, { 1, 3, 4 }, { 2, 4, 5 } };
	return data[1 + map2[i][j]];
}

template<class T>
CUDA_EXPORT inline T& multipole_type<T>::operator ()(int i, int j) {
	static constexpr size_t map2[3][3] = { { 0, 1, 2 }, { 1, 3, 4 }, { 2, 4, 5 } };
	return data[1 + map2[i][j]];
}

template<class T>
CUDA_EXPORT inline T multipole_type<T>::operator ()(int i, int j, int k) const {
	static constexpr size_t map3[3][3][3] = { { { 0, 1, 2 }, { 1, 3, 4 }, { 2, 4, 5 } }, { { 1, 3, 4 }, { 3, 6, 7 }, { 4,
			7, 8 } }, { { 2, 4, 5 }, { 4, 7, 8 }, { 5, 8, 9 } } };

	return data[7 + map3[i][j][k]];
}
template<class T>
CUDA_EXPORT inline T& multipole_type<T>::operator ()(int i, int j, int k) {
	static constexpr size_t map3[3][3][3] = { { { 0, 1, 2 }, { 1, 3, 4 }, { 2, 4, 5 } }, { { 1, 3, 4 }, { 3, 6, 7 }, { 4,
			7, 8 } }, { { 2, 4, 5 }, { 4, 7, 8 }, { 5, 8, 9 } } };
	return data[7 + map3[i][j][k]];
}

template<class T>
CUDA_EXPORT inline multipole_type<T>& multipole_type<T>::operator =(const multipole_type<T> &other) {
	memcpy(&data[0], &other.data[0], MP * sizeof(float));
	return *this;
}

template<class T>
CUDA_EXPORT inline multipole_type<T>& multipole_type<T>::operator =(T other) {
	for (int i = 0; i < MP; i++) {
		data[i] = other;
	}
	return *this;
}

template<class T>
template<class V>
CUDA_EXPORT inline multipole_type<T> multipole_type<T>::operator>>(const std::array<V, NDIM> &dX) const {
	multipole_type you = *this;
	you >>= dX;
	return you;
}

template<class T>
CUDA_EXPORT inline multipole_type<T> multipole_type<T>::operator +(const multipole_type<T> &vec) const {
	multipole_type<T> C;
	for (int i = 0; i < MP; i++) {
		C[i] = data[i] + vec[i];
	}
	return C;
}

template<class T>
template<class V>
CUDA_EXPORT inline multipole_type<T>& multipole_type<T>::operator>>=(const std::array<V, NDIM> &X) {
	multipole_type<T>& M = *this;
	const auto X00 = X[0] * X[0];
	const auto X01 = X[0] * X[1];
	const auto X02 = X[0] * X[2];
	const auto X11 = X[1] * X[1];
	const auto X12 = X[1] * X[2];
	const auto X22 = X[2] * X[2];
	M[7] -= M[0] * X00 * X[0];
	M[7] -= T(3) * M[1] * X[0];
	M[8] -= M[0] * X00 * X[1];
	M[8] -= M[1] * X[1];
	M[8] -= T(2) * M[2] * X[0];
	M[9] -= M[0] * X00 * X[2];
	M[9] -= M[1] * X[2];
	M[9] -= T(2) * M[3] * X[0];
	M[10] -= M[0] * X01 * X[1];
	M[10] -= T(2) * M[2] * X[1];
	M[10] -= M[4] * X[0];
	M[11] -= M[0] * X01 * X[2];
	M[11] -= M[2] * X[2];
	M[11] -= M[5] * X[0];
	M[11] -= M[3] * X[1];
	M[12] -= M[0] * X02 * X[2];
	M[12] -= T(2) * M[3] * X[2];
	M[12] -= M[6] * X[0];
	M[13] -= M[0] * X11 * X[1];
	M[13] -= T(3) * M[4] * X[1];
	M[14] -= M[0] * X11 * X[2];
	M[14] -= M[4] * X[2];
	M[14] -= T(2) * M[5] * X[1];
	M[15] -= M[0] * X12 * X[2];
	M[15] -= T(2) * M[5] * X[2];
	M[15] -= M[6] * X[1];
	M[16] -= M[0] * X22 * X[2];
	M[16] -= T(3) * M[6] * X[2];
	M[1] = fma(M[0], X00, M[1]);
	M[2] = fma(M[0], X01, M[2]);
	M[3] = fma(M[0], X02, M[3]);
	M[4] = fma(M[0], X11, M[4]);
	M[5] = fma(M[0], X12, M[5]);
	M[6] = fma(M[0], X22, M[6]);
	return M;
}

using multipole = multipole_type<float>;

struct multi_source {
	multipole m;
	std::array<fixed32, NDIM> x;
};

#endif /* multipole_type_H_ */

