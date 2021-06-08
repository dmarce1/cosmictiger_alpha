/*
 * array.hpp
 *
 *  Created on: Feb 3, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_ARRAY_HPP_
#define COSMICTIGER_ARRAY_HPP_

#include <cassert>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/cuda.hpp>

template<class T, int N>
class array {
	T ptr[N];
public:
	CUDA_EXPORT
	const T& operator[](int i) const {
		BOUNDS_CHECK1(i, 0, N);
		return ptr[i];
	}
	CUDA_EXPORT
	T& operator[](int i) {
		BOUNDS_CHECK1(i, 0, N);
		return ptr[i];
	}

	CUDA_EXPORT
	T* data() {
		return ptr;
	}

	CUDA_EXPORT
	const T* data() const {
		return ptr;
	}

	CUDA_EXPORT
	T* begin() {
		return ptr;
	}

	CUDA_EXPORT
	T* end() {
		return ptr + N;
	}

	CUDA_EXPORT
	inline array<T, N> operator+(const array<T, N>& other) {
		array<T, N> result;
		for (int i = 0; i < N; i++) {
			result[i] = (*this)[i] + other[i];
		}
		return result;
	}

	CUDA_EXPORT
	inline array<T, N> operator-(const array<T, N>& other) {
		array<T, N> result;
		for (int i = 0; i < N; i++) {
			result[i] = (*this)[i] - other[i];
		}
		return result;
	}

	CUDA_EXPORT
	inline array<T, N> operator*(const T& other) {
		array<T, N> result;
		for (int i = 0; i < N; i++) {
			result[i] = (*this)[i] * other;
		}
		return result;
	}

	CUDA_EXPORT
	inline array<T, N> operator-() const {
		array<T, N> result;
		for (int i = 0; i < N; i++) {
			result[i] = -(*this)[i];
		}
		return result;
	}

	template<class A>
	void serialize(A&& arc, unsigned) {
		for (int i = 0; i < N; i++) {
			arc & (*this)[i];
		}
	}

};



#endif /* COSMICTIGER_ARRAY_HPP_ */
