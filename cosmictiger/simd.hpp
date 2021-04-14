/*
 * simd.hpp
 *
 *  Created on: Jul 3, 2020
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_SIMD_HPP_
#define COSMICTIGER_SIMD_HPP_

#include <cosmictiger/defs.hpp>

#include <immintrin.h>

#include <cmath>

#define USE_AVX2

#ifdef USE_AVX2
#define SIMDALIGN                  __attribute__((aligned(32)))
#define SIMD_VLEN 8
#define _simd_float                 __m256
#define _simd_int                   __m256i
#define _mmx_add_ps(a,b)            _mm256_add_ps((a),(b))
#define _mmx_sub_ps(a,b)            _mm256_sub_ps((a),(b))
#define _mmx_mul_ps(a,b)            _mm256_mul_ps((a),(b))
#define _mmx_div_ps(a,b)            _mm256_div_ps((a),(b))
#define _mmx_add_pd(a,b)            _mm256_add_pd((a),(b))
#define _mmx_sub_pd(a,b)            _mm256_sub_pd((a),(b))
#define _mmx_mul_pd(a,b)            _mm256_mul_pd((a),(b))
#define _mmx_div_pd(a,b)            _mm256_div_pd((a),(b))
#define _mmx_sqrt_ps(a)             _mm256_sqrt_ps(a)
#define _mmx_min_ps(a, b)           _mm256_min_ps((a),(b))
#define _mmx_max_ps(a, b)           _mm256_max_ps((a),(b))
#define _mmx_max_pd(a, b)           _mm256_max_pd((a),(b))
#define _mmx_or_ps(a, b)            _mm256_or_ps((a),(b))
#define _mmx_and_ps(a, b)           _mm256_and_ps((a),(b))
#define _mmx_andnot_ps(a, b)        _mm256_andnot_ps((a),(b))
#define _mmx_rsqrt_ps(a)            _mm256_rsqrt_ps(a)
#define _mmx_add_epi32(a,b)         _mm256_add_epi32((a),(b))
#define _mmx_sub_epi32(a,b)         _mm256_sub_epi32((a),(b))
#define _mmx_mul_epi32(a,b)         _mm256_mullo_epi32((a),(b))
#define _mmx_cvtps_epi32(a)         _mm256_cvtps_epi32((a))
#define _mmx_fmadd_ps(a,b,c)        _mm256_fmadd_ps ((a),(b),(c))
#define _mmx_cmp_ps(a,b,c)        	_mm256_cmp_ps(a,b,c)
#endif

class simd_float;
class simd_int;

class simd_float {
private:
	union {
		_simd_float v;
		float floats[8];
	};
public:
	static constexpr std::size_t size() {
		return SIMD_VLEN;
	}
	simd_float() = default;
	inline ~simd_float() = default;
	simd_float(const simd_float&) = default;
	inline simd_float(float d) {
		v = _mm256_set_ps(d, d, d, d, d, d, d, d);
	}
	inline simd_float(float d0, float d1, float d2, float d3, float d4, float d5, float d6, float d7) {
		v = _mm256_set_ps(d7, d6, d5, d4, d3, d2, d1, d0);
	}
	inline float sum() const {
		float sum = 0.0f;
		for (int i = 0; i < size(); i++) {
			sum += (*this)[i];
		}
		return sum;
	}

	inline simd_float(simd_int i);

	inline simd_float& operator=(const simd_float &other) = default;
	simd_float& operator=(simd_float &&other) = default;
	inline simd_float operator+(const simd_float &other) const {
		simd_float r;
		r.v = _mmx_add_ps(v, other.v);
		return r;
	}
	inline simd_float operator-(const simd_float &other) const {
		simd_float r;
		r.v = _mmx_sub_ps(v, other.v);
		return r;
	}
	inline simd_float operator*(const simd_float &other) const {
		simd_float r;
		r.v = _mmx_mul_ps(v, other.v);
		return r;
	}
	inline simd_float operator/(const simd_float &other) const {
		simd_float r;
		r.v = _mmx_div_ps(v, other.v);
		return r;
	}
	inline simd_float operator+() const {
		return *this;
	}
	inline simd_float operator-() const {
		return simd_float(0.0) - *this;
	}
	inline simd_float& operator+=(const simd_float &other) {
		v = _mmx_add_ps(v, other.v);
		return *this;
	}
	inline simd_float& operator-=(const simd_float &other) {
		v = _mmx_sub_ps(v, other.v);
		return *this;
	}
	inline simd_float& operator*=(const simd_float &other) {
		v = _mmx_mul_ps(v, other.v);
		return *this;
	}
	inline simd_float& operator/=(const simd_float &other) {
		v = _mmx_div_ps(v, other.v);
		return *this;
	}

	inline simd_float operator*(float d) const {
		const simd_float other = d;
		return other * *this;
	}
	inline simd_float operator/(float d) const {
		const simd_float other = 1.0 / d;
		return *this * other;
	}

	inline simd_float operator*=(float d) {
		*this = *this * d;
		return *this;
	}
	inline simd_float operator/=(float d) {
		*this = *this * (1.0 / d);
		return *this;
	}
	inline float& operator[](std::size_t i) {
		return floats[i];
	}
	inline float operator[](std::size_t i) const {
		return floats[i];
	}
	friend bool isinf(const simd_float&);
	friend bool isnan(const simd_float&);
	friend simd_float copysign(const simd_float&, const simd_float&);
	friend simd_float sqrt(const simd_float&);
	friend simd_float rsqrt(const simd_float&);
	friend simd_float operator*(float, const simd_float &other);
	friend simd_float operator/(float, const simd_float &other);
	friend simd_float max(const simd_float &a, const simd_float &b);
	friend simd_float min(const simd_float &a, const simd_float &b);
	friend simd_float fma(const simd_float &a, const simd_float &b, const simd_float &c);
	friend simd_float round(const simd_float);

	friend simd_float sin(const simd_float &a);
	friend simd_float cos(const simd_float &a);
	friend simd_float abs(const simd_float &a);
	friend simd_float erfexp(const simd_float &a, simd_float*);
	friend simd_float gather(void*, int, int);

	friend simd_float two_pow(const simd_float &r);
	friend void sincos(const simd_float &x, simd_float *s, simd_float *c);
	simd_float operator<(simd_float other) const { // 2
		simd_float rc;
		static const simd_float one(1);
		static const simd_float zero(0);
		auto mask0 = _mmx_cmp_ps(v, other.v, _CMP_LT_OQ);
		rc.v = _mmx_and_ps(mask0, one.v);
		return rc;
	}
	simd_float operator>(simd_float other) const { // 2
		simd_float rc;
		static const simd_float one(1);
		static const simd_float zero(0);
		auto mask0 = _mmx_cmp_ps(v, other.v, _CMP_GT_OQ);
		rc.v = _mmx_and_ps(mask0, one.v);
		return rc;
	}

	friend class simd_int;
}SIMDALIGN;
;

class simd_int {
private:
	union {
		_simd_int v;
		int ints[8];
	};
public:
	static constexpr std::size_t size() {
		return SIMD_VLEN;
	}
	simd_int() = default;
	inline ~simd_int() = default;
	simd_int(const simd_int&) = default;
	simd_int(simd_float r) {
		v = _mm256_cvtps_epi32(_mm256_floor_ps(r.v));
	}
	inline simd_int(int d) {
		v = _mm256_set_epi32(d, d, d, d, d, d, d, d);
	}
	inline simd_int& operator=(const simd_int &other) = default;
	simd_int& operator=(simd_int &&other) = default;
	inline simd_int operator+(const simd_int &other) const {
		simd_int r;
		r.v = _mmx_add_epi32(v, other.v);
		return r;
	}
	inline simd_int operator-(const simd_int &other) const {
		simd_int r;
		r.v = _mmx_sub_epi32(v, other.v);
		return r;
	}
	inline simd_int operator*(const simd_int &other) const {
		simd_int r;
		r.v = _mmx_mul_epi32(v, other.v);
		return r;
	}
	inline simd_int operator+() const {
		return *this;
	}
	inline simd_int operator-() const {
		return simd_int(0.0) - *this;
	}
	inline simd_int operator/(int d) const {
		const simd_int other = 1.0 / d;
		return *this * other;
	}
	inline int& operator[](std::size_t i) {
		return ints[i];
	}
	inline int operator[](std::size_t i) const {
		return ints[i];
	}
	inline int sum() const {
		int sum = 0;
		for (int i = 0; i < simd_int::size(); i++) {
			sum += ints[i];
		}
		return sum;
	}
	friend class simd_float;

}SIMDALIGN;

inline simd_float::simd_float(simd_int i) {
	v = _mm256_cvtepi32_ps(i.v);

}
inline simd_float two_pow(const simd_float &r) {											// 21
	static const simd_float zero = simd_float(0.0);
	static const simd_float one = simd_float(1.0);
	static const simd_float c1 = simd_float(std::log(2));
	static const simd_float c2 = simd_float((0.5) * std::pow(std::log(2), 2));
	static const simd_float c3 = simd_float((1.0 / 6.0) * std::pow(std::log(2), 3));
	static const simd_float c4 = simd_float((1.0 / 24.0) * std::pow(std::log(2), 4));
	static const simd_float c5 = simd_float((1.0 / 120.0) * std::pow(std::log(2), 5));
	static const simd_float c6 = simd_float((1.0 / 720.0) * std::pow(std::log(2), 6));
	static const simd_float c7 = simd_float((1.0 / 5040.0) * std::pow(std::log(2), 7));
	static const simd_float c8 = simd_float((1.0 / 40320.0) * std::pow(std::log(2), 8));
	simd_float r0;
	__m256i n;
	r0.v = _mm256_round_ps(r.v, _MM_FROUND_TO_NEAREST_INT);							// 1
	n = _mm256_cvtps_epi32(r0.v);														// 1
	auto x = r - r0;
	auto y = c8;
	y = fma(y, x, c7);																		// 2
	y = fma(y, x, c6);																		// 2
	y = fma(y, x, c5);																		// 2
	y = fma(y, x, c4);																		// 2
	y = fma(y, x, c3);																		// 2
	y = fma(y, x, c2);																		// 2
	y = fma(y, x, c1);																		// 2
	y = fma(y, x, one);																		// 2
	static const auto sevenf = _mm256_set_epi32(0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f);
	auto imm00 = _mm256_add_epi32(n, sevenf);
	imm00 = _mm256_slli_epi32(imm00, 23);
	r0.v = _mm256_castsi256_ps(imm00);
	auto res = y * r0;																			// 1
	return res;
}

inline simd_float round(const simd_float a) {
	simd_float v;
	v.v = _mm256_round_ps(a.v, _MM_FROUND_TO_NEAREST_INT);
	return v;
}

inline simd_float sin(const simd_float &x0) {						// 17
	auto x = x0;
	// From : http://mooooo.ooo/chebyshev-sine-approximation/
	static const simd_float pi_major(3.1415927);
	static const simd_float pi_minor(-0.00000008742278);
	x = x - round(x * (1.0 / (2.0 * M_PI))) * (2.0 * M_PI);			// 4
	const simd_float x2 = x * x;									// 1
	simd_float p = simd_float(0.00000000013291342);
	p = fma(p, x2, simd_float(-0.000000023317787));				// 2
	p = fma(p, x2, simd_float(0.0000025222919));					// 2
	p = fma(p, x2, simd_float(-0.00017350505));					// 2
	p = fma(p, x2, simd_float(0.0066208798));						// 2
	p = fma(p, x2, simd_float(-0.10132118));						// 2
	const auto x1 = (x - pi_major - pi_minor);						// 2
	const auto x3 = (x + pi_major + pi_minor);						// 2
	auto res = x1 * x3 * p * x;										// 3
	return res;
}

inline simd_float cos(const simd_float &x) {		// 18
	return sin(x + simd_float(M_PI / 2.0));
}

inline void sincos(const simd_float &x, simd_float *s, simd_float *c) {		// 35
//#ifdef __AVX512F__
//	s->v = _mm512_sincos_ps(&(c->v),x.v);
//#else
	*s = sin(x);
	*c = cos(x);
//#endif
}

inline simd_float exp(simd_float a) { 	// 24
	static const simd_float c0 = 1.0 / std::log(2);
	static const auto hi = simd_float(88);
	static const auto lo = simd_float(-88);
	a = min(a, hi);
	a = max(a, lo);
	return two_pow(a * c0);
}

inline simd_float erfcexp(const simd_float &x, simd_float *e) {				// 76
	const simd_float p(0.3275911);
	const simd_float a1(0.254829592);
	const simd_float a2(-0.284496736);
	const simd_float a3(1.421413741);
	const simd_float a4(-1.453152027);
	const simd_float a5(1.061405429);
	const simd_float t1 = simd_float(1) / (simd_float(1) + p * x);			//37
	const simd_float t2 = t1 * t1;											// 1
	const simd_float t3 = t2 * t1;											// 1
	const simd_float t4 = t2 * t2;											// 1
	const simd_float t5 = t2 * t3;											// 1
	*e = exp(-x * x);														// 16
	return (a1 * t1 + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * *e; 			// 11
}

inline simd_float fma(const simd_float &a, const simd_float &b, const simd_float &c) {

	simd_float v;
	v.v = _mmx_fmadd_ps(a.v, b.v, c.v);
	return v;
}

inline simd_float fmaf(const simd_float &a, const simd_float &b, const simd_float &c) {
	return fma(a, b, c);
}

inline simd_float sqrt(const simd_float &vec) {
	simd_float r;
	r.v = _mmx_sqrt_ps(vec.v);
	return r;

}

inline simd_float rsqrt(const simd_float &vec) {
	simd_float r;
	r.v = _mmx_rsqrt_ps(vec.v);
	return r;

}

inline simd_float operator*(float d, const simd_float &other) {
	const simd_float a = d;
	return a * other;
}

inline simd_float operator/(float d, const simd_float &other) {
	const simd_float a = d;
	return a / other;
}
inline simd_float min(const simd_float &a, const simd_float &b) {
	simd_float r;
	r.v = _mmx_min_ps(a.v, b.v);
	return r;
}

inline simd_float copysign(const simd_float &y, const simd_float &x) {
	simd_float v;
	constexpr float signbit = -0.f;
	static simd_float const avx_signbit = simd_float(signbit);
	const auto tmp0 = _mmx_andnot_ps(avx_signbit.v, y.v);
	const auto tmp2 = _mmx_and_ps(avx_signbit.v, x.v);
	v.v = _mmx_or_ps(tmp2, tmp0); // (avx_signbit & from) | (~avx_signbit & to)
	return v;
}

inline simd_float abs(const simd_float &a) {
	return max(a, -a);
}

inline simd_float max(const simd_float &a, const simd_float &b) {
	simd_float r;
	r.v = _mmx_max_ps(a.v, b.v);
	return r;
}

inline bool isinf(const simd_float& f) {
	for (int i = 0; i < simd_float::size(); i++) {
		if (std::isinf(f[i])) {
			return true;
		}
	}
	return false;
}

inline bool isnan(const simd_float& f) {
	for (int i = 0; i < simd_float::size(); i++) {
		if (std::isnan(f[i])) {
			return true;
		}
	}
	return false;
}

#endif /* COSMICTIGER_SIMD_HPP_ */
