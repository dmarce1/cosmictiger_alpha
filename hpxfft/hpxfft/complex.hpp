/*
 * complex.hpp
 *
 *  Created on: Jun 20, 2021
 *      Author: dmarce1
 */

#ifndef COMPLEX_HPP_
#define COMPLEX_HPP_

#include <hpxfft/cuda.hpp>



template<class T = float>
class complex {
	T x, y;
public:
	complex() = default;
	CUDA_EXPORT
	complex(float a) {
		x = a;
		y = T(0.0);
	}
	CUDA_EXPORT
	complex(float a, float b) {
		x = a;
		y = b;
	}
	CUDA_EXPORT
	complex& operator+=(complex other) {
		x += other.x;
		y += other.y;
		return *this;
	}
	CUDA_EXPORT
	complex& operator-=(complex other) {
		x -= other.x;
		y -= other.y;
		return *this;
	}
	CUDA_EXPORT
	complex operator*(complex other) const {
		complex a;
		a.x = x * other.x - y * other.y;
		a.y = x * other.y + y * other.x;
		return a;
	}
	CUDA_EXPORT
	complex operator/(complex other) const {
		return *this * other.conj() / other.norm();
	}
	CUDA_EXPORT
	complex operator/(float other) const {
		complex b;
		b.x = x / other;
		b.y = y / other;
		return b;
	}
	CUDA_EXPORT
	complex operator*(float other) const {
		complex b;
		b.x = x * other;
		b.y = y * other;
		return b;
	}
	CUDA_EXPORT
	complex operator+(complex other) const {
		complex a;
		a.x = x + other.x;
		a.y = y + other.y;
		return a;
	}
	CUDA_EXPORT
	complex operator-(complex other) const {
		complex a;
		a.x = x - other.x;
		a.y = y - other.y;
		return a;
	}
	CUDA_EXPORT
	complex conj() const {
		complex a;
		a.x = x;
		a.y = -y;
		return a;
	}
	CUDA_EXPORT
	float real() const {
		return x;
	}
	CUDA_EXPORT
	float imag() const {
		return y;
	}
	CUDA_EXPORT
	float& real() {
		return x;
	}
	CUDA_EXPORT
	float& imag() {
		return y;
	}
	CUDA_EXPORT
	float norm() const {
		return ((*this) * conj()).real();
	}
	CUDA_EXPORT
	float abs() const {
		return sqrtf(norm());
	}
	CUDA_EXPORT
	complex operator-() const {
		complex a;
		a.x = -x;
		a.y = -y;
		return a;
	}
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & x;
		arc & y;
	}
};

template<class T>
inline void swap(complex<T>& a, complex<T>& b) {
	std::swap(a.real(), b.real());
	std::swap(a.imag(), b.imag());
}

template<class T>
CUDA_EXPORT inline complex<T> operator*(T a, complex<T> b) {
	return b * a;
}

template<class T>
CUDA_EXPORT inline complex<T> expc(complex<T> z) {
	float x, y;
	float t = EXP(z.real());
	sincosf(z.imag(), &y, &x);
	x *= t;
	y *= t;
	return complex<T>(x, y);
}


using cmplx = complex<float>;

#endif /* COMPLEX_HPP_ */
