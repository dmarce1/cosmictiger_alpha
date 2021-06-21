/*
 * fourier.hpp
 *
 *  Created on: Jan 12, 2021
 *      Author: dmarce1
 */

#ifndef FOURIER_HPP_
#define FOURIER_HPP_



#include <vector>
#include <hpxfft/cuda.hpp>



template<class T>
CUDA_EXPORT inline T sqr(T a) {
	return a * a;
}
template<class T = float>
class complex {
	T x, y;
public:
	complex() = default;
	CUDA_EXPORT complex(float a) {
		x = a;
		y = T(0.0);
	}
	CUDA_EXPORT complex(float a, float b) {
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

namespace hpxfft {
int hpx_rank();
}

void fft1d(std::vector<cmplx>& Y, int N);
void fft2d(std::vector<cmplx>& Y, int N);

void fourier3d_initialize(int N_);
void fourier3d_destroy();
void fourier3d_execute();
void fourier3d_inv_execute();
void fourier3d_accumulate_real(int xb, int xe, int yb, int ye, int zb, int ze, std::vector<float> data);
void fourier3d_accumulate(int xb, int xe, int yb, int ye, int zb, int ze, std::vector<cmplx> data);
std::vector<cmplx> fourier3d_read(int xb, int xe, int yb, int ye, int zb, int ze);
std::vector<float> fourier3d_read_real(int xb, int xe, int yb, int ye, int zb, int ze);
void fourier3d_mirror();
std::vector<float> fourier3d_power_spectrum();

#endif /* FOURIER_HPP_ */
