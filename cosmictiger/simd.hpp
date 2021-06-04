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

#ifdef __AVX2__
#define USE_AVX2
#elif defined(__AVX__)
#define USE_AVX
#else
#define USE_SCALAR
#endif

#define SIMD_VLEN 8

#define SIMDALIGN                  __attribute__((aligned(32)))

#ifdef USE_AVX2
#define SIMD_N    1
#define _simd_float                 __m256
#define _simd_int                   __m256i
#define mmx_add_ps(a,b)            _mm256_add_ps((a),(b))
#define mmx_sub_ps(a,b)            _mm256_sub_ps((a),(b))
#define mmx_mul_ps(a,b)            _mm256_mul_ps((a),(b))
#define mmx_div_ps(a,b)            _mm256_div_ps((a),(b))
#define mmx_sqrt_ps(a)             _mm256_sqrt_ps(a)
#define mmx_min_ps(a, b)           _mm256_min_ps((a),(b))
#define mmx_max_ps(a, b)           _mm256_max_ps((a),(b))
#define mmx_or_ps(a, b)            _mm256_or_ps((a),(b))
#define mmx_and_ps(a, b)           _mm256_and_ps((a),(b))
#define mmx_andnot_ps(a, b)        _mm256_andnot_ps((a),(b))
#define mmx_rsqrt_ps(a)            _mm256_rsqrt_ps(a)
#define mmx_add_epi32(a,b)         _mm256_add_epi32((a),(b))
#define mmx_sub_epi32(a,b)         _mm256_sub_epi32((a),(b))
#define mmx_mul_epi32(a,b)         _mm256_mullo_epi32((a),(b))
#define mmx_cvtepi32_ps(a)         _mm256_cvtepi32_ps((a))
#define mmx_cvtps_epi32(a)         _mm256_cvtps_epi32((a))
#define mmx_fmadd_ps(a,b,c)        _mm256_fmadd_ps ((a),(b),(c))
#define mmx_cmp_ps(a,b,c)          _mm256_cmp_ps(a,b,c)
#define mmx_round_ps(a,b)			  _mm256_round_ps(a,b)
#define mmx_slli_epi32(a,b)	     _mm256_slli_epi32(a,b)
#define mmx_floor_ps(a)            _mm256_floor_ps(a)
#define mmx_castsi_ps(a)        _mm256_castsi256_ps(a)
#elif defined( USE_AVX)
#define SIMD_N    2
#define _simd_float                 __m128
#define _simd_int                   __m128i
#define mmx_add_ps(a,b)            _mm_add_ps((a),(b))
#define mmx_sub_ps(a,b)            _mm_sub_ps((a),(b))
#define mmx_mul_ps(a,b)            _mm_mul_ps((a),(b))
#define mmx_div_ps(a,b)            _mm_div_ps((a),(b))
#define mmx_sqrt_ps(a)             _mm_sqrt_ps(a)
#define mmx_min_ps(a, b)           _mm_min_ps((a),(b))
#define mmx_max_ps(a, b)           _mm_max_ps((a),(b))
#define mmx_or_ps(a, b)            _mm_or_ps((a),(b))
#define mmx_and_ps(a, b)           _mm_and_ps((a),(b))
#define mmx_andnot_ps(a, b)        _mm_andnot_ps((a),(b))
#define mmx_rsqrt_ps(a)            _mm_rsqrt_ps(a)
#define mmx_add_epi32(a,b)         _mm_add_epi32((a),(b))
#define mmx_sub_epi32(a,b)         _mm_sub_epi32((a),(b))
#define mmx_mul_epi32(a,b)         _mm_mullo_epi32((a),(b))
#define mmx_cvtepi32_ps(a)         _mm_cvtepi32_ps((a))
#define mmx_cvtps_epi32(a)         _mm_cvtps_epi32((a))
#define mmx_fmadd_ps(a,b,c)        _mm_add_ps(_mm_mul_ps ((a),(b)), (c))
#define mmx_cmp_ps(a,b,c)          _mm_cmp_ps(a,b,c)
#define mmx_round_ps(a,b)	   _mm_round_ps(a,b)
#define mmx_slli_epi32(a,b)	   _mm_slli_epi32(a,b)
#define mmx_floor_ps(a)            _mm_floor_ps(a)
#define mmx_castsi_ps(a)           _mm_castsi128_ps(a)
#elif defined(USE_SCALAR)
#define SIMD_N    8
#define _simd_float                 float
#define _simd_int                   int
#define mmx_add_ps(a,b)            (a+b)
#define mmx_sub_ps(a,b)            (a-b)
#define mmx_mul_ps(a,b)            (a*b)
#define mmx_div_ps(a,b)            (a/b)
#define mmx_sqrt_ps(a)             std::sqrt(a)
#define mmx_min_ps(a, b)           std::min(a,b)
#define mmx_max_ps(a, b)           std::max(a,b)
#define mmx_rsqrt_ps(a)            (1.0f / std::sqrt(a))
#define mmx_add_epi32(a,b)         (a+b)
#define mmx_sub_epi32(a,b)         (a-b)
#define mmx_mul_epi32(a,b)         (a*b)
#define mmx_cvtepi32_ps(a)         float(a)
#define mmx_cvtps_epi32(a)         int(a)
#define mmx_fmadd_ps(a,b,c)        (a*b+c)
#define mmx_slli_epi32(a,b)	     (a<<b)
#define mmx_floor_ps(a)            std::floor(a)
#endif

class simd_float;
class simd_int;

class simd_float {
private:
	union {
		_simd_float v[SIMD_N];
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
#ifdef USE_AVX2
		v[0] = _mm256_set_ps(d, d, d, d, d, d, d, d);
#elif defined(USE_AVX)
		for( int i = 0; i < SIMD_N; i++) {
			v[i] = _mm_set_ps(d, d, d, d);
		}
#else
		for (int i = 0; i < SIMD_N; i++) {
			v[i] = d;
		}
#endif
	}
	inline simd_float(float d0, float d1, float d2, float d3, float d4, float d5, float d6, float d7) {
#ifdef USE_AVX2
		v[0] = _mm256_set_ps(d7, d6, d5, d4, d3, d2, d1, d0);
#elif defined(USE_AVX)
		v[0] = _mm_set_ps(d3, d2, d1, d0);
		v[1] = _mm_set_ps(d7, d6, d5, d4);
#else
		v[0] = d0;
		v[1] = d1;
		v[2] = d2;
		v[3] = d3;
		v[4] = d4;
		v[5] = d5;
		v[6] = d6;
		v[7] = d7;
#endif
	}
	inline float sum() const {
#ifdef USE_AVX2
		__m128 vsum;
		__m128& a = *(((__m128*) v) + 0);
		__m128& b = *(((__m128*) v) + 1);
		vsum = _mm_add_ps(a, b);
		return vsum[0] + vsum[1] + vsum[2] + vsum[3];
#else
		float sum = (*this)[0];
		for (int i = 1; i < size(); i++) {
			sum += (*this)[i];
		}
		return sum;
#endif
	}

	inline simd_float(simd_int i);

	inline simd_float& operator=(const simd_float &other) = default;
	simd_float& operator=(simd_float &&other) = default;
	inline simd_float operator+(const simd_float &other) const {
		simd_float r;
		for (int i = 0; i < SIMD_N; i++) {
			r.v[i] = mmx_add_ps(v[i], other.v[i]);
		}
		return r;
	}
	inline simd_float operator-(const simd_float &other) const {
		simd_float r;
		for (int i = 0; i < SIMD_N; i++) {
			r.v[i] = mmx_sub_ps(v[i], other.v[i]);
		}
		return r;
	}
	inline simd_float operator*(const simd_float &other) const {
		simd_float r;
		for (int i = 0; i < SIMD_N; i++) {
			r.v[i] = mmx_mul_ps(v[i], other.v[i]);
		}
		return r;
	}
	inline simd_float operator/(const simd_float &other) const {
		simd_float r;
		for (int i = 0; i < SIMD_N; i++) {
			r.v[i] = mmx_div_ps(v[i], other.v[i]);
		}
		return r;
	}
	inline simd_float operator+() const {
		return *this;
	}
	inline simd_float operator-() const {
		return simd_float(0.0) - *this;
	}
	inline simd_float& operator+=(const simd_float &other) {
		for (int i = 0; i < SIMD_N; i++) {
			v[i] = mmx_add_ps(v[i], other.v[i]);
		}
		return *this;
	}
	inline simd_float& operator-=(const simd_float &other) {
		for (int i = 0; i < SIMD_N; i++) {
			v[i] = mmx_sub_ps(v[i], other.v[i]);
		}
		return *this;
	}
	inline simd_float& operator*=(const simd_float &other) {
		for (int i = 0; i < SIMD_N; i++) {
			v[i] = mmx_mul_ps(v[i], other.v[i]);
		}
		return *this;
	}
	inline simd_float& operator/=(const simd_float &other) {
		for (int i = 0; i < SIMD_N; i++) {
			v[i] = mmx_div_ps(v[i], other.v[i]);
		}
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
		for (int i = 0; i < SIMD_N; i++) {
#ifdef USE_SCALAR
			rc.v[i] = v[i] < other.v[i];
#else
			auto mask0 = mmx_cmp_ps(v[i], other.v[i], _CMP_LT_OQ);
			rc.v[i] = mmx_and_ps(mask0, one.v[i]);
#endif
		}
		return rc;
	}
	simd_float operator>(simd_float other) const { // 2
		simd_float rc;
		static const simd_float one(1);
		static const simd_float zero(0);
		for (int i = 0; i < SIMD_N; i++) {
#ifdef USE_SCALAR
			rc.v[i] = v[i] > other.v[i];
#else
			auto mask0 = mmx_cmp_ps(v[i], other.v[i], _CMP_GT_OQ);
			rc.v[i] = mmx_and_ps(mask0, one.v[i]);
#endif
		}
		return rc;
	}

	friend class simd_int;
}SIMDALIGN;
;

class simd_int {
private:
	union {
		_simd_int v[SIMD_N];
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
		for (int i = 0; i < SIMD_N; i++) {
			v[i] = mmx_cvtps_epi32(mmx_floor_ps(r.v[i]));
		}
	}
	inline simd_int(int d) {
#ifdef USE_AVX2
		v[0] = _mm256_set_epi32(d, d, d, d, d, d, d, d);
#elif defined(USE_AVX)
		for( int i = 0; i < SIMD_N; i++) {
			v[i] = _mm_set_epi32(d, d, d, d);
		}
#else
		for (int i = 0; i < SIMD_N; i++) {
			v[i] = d;
		}
#endif
	}
	inline simd_int& operator=(const simd_int &other) = default;
	simd_int& operator=(simd_int &&other) = default;
	inline simd_int operator+(const simd_int &other) const {
		simd_int r;
		for (int i = 0; i < SIMD_N; i++) {
			r.v[i] = mmx_add_epi32(v[i], other.v[i]);
		}
		return r;
	}
	inline simd_int operator-(const simd_int &other) const {
		simd_int r;
		for (int i = 0; i < SIMD_N; i++) {
			r.v[i] = mmx_sub_epi32(v[i], other.v[i]);
		}
		return r;
	}
	inline simd_int operator*(const simd_int &other) const {
		simd_int r;
		for (int i = 0; i < SIMD_N; i++) {
			r.v[i] = mmx_mul_epi32(v[i], other.v[i]);
		}
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
	inline simd_int operator<<=(int shft) {
		for (int i = 0; i < SIMD_N; i++) {
			v[i] = mmx_slli_epi32(v[i], shft);
		}
		return *this;
	}
	simd_float cast2float() const {
#ifdef USE_SCALAR
		simd_float r;
		for (int i = 0; i < SIMD_N; i++) {
			r.v[i] = *((float*) (v + i));
		}
		return r;

#else
		simd_float r;
		for( int i = 0; i < SIMD_N; i++) {
			r.v[i] = mmx_castsi_ps(v[i]);
		}
		return r;
#endif
	}
	friend class simd_float;

}SIMDALIGN;

inline simd_float::simd_float(simd_int ints) {
	for (int i = 0; i < SIMD_N; i++) {
		v[i] = mmx_cvtepi32_ps(ints.v[i]);
	}
}

inline simd_float round(const simd_float a) {
	simd_float v;
	for (int i = 0; i < SIMD_N; i++) {
#ifdef USE_SCALAR
		v.v[i] = std::round(a.v[i]);
#else
		v.v[i] = mmx_round_ps(a.v[i], _MM_FROUND_TO_NEAREST_INT);
#endif
	}
	return v;
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
	simd_int n;
	r0 = round(r);							// 1
	n = simd_int(r0);														// 1
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
	simd_int sevenf(0x7f);
	simd_int imm00 = n + sevenf;
	imm00 <<= 23;
	r0 = imm00.cast2float();
	auto res = y * r0;																			// 1
	return res;
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
	for (int i = 0; i < SIMD_N; i++) {
		v.v[i] = mmx_fmadd_ps(a.v[i], b.v[i], c.v[i]);
	}
	return v;
}

inline simd_float fmaf(const simd_float &a, const simd_float &b, const simd_float &c) {
	return fma(a, b, c);
}

inline simd_float sqrt(const simd_float &vec) {
	simd_float r;
	for (int i = 0; i < SIMD_N; i++) {
		r.v[i] = mmx_sqrt_ps(vec.v[i]);
	}
	return r;

}

inline simd_float rsqrt(const simd_float &vec) {
	simd_float r;
	for (int i = 0; i < SIMD_N; i++) {
		r.v[i] = mmx_rsqrt_ps(vec.v[i]);
	}
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
	for (int i = 0; i < SIMD_N; i++) {
		r.v[i] = mmx_min_ps(a.v[i], b.v[i]);
	}
	return r;
}

inline simd_float copysign(const simd_float &y, const simd_float &x) {
	simd_float v;
#ifdef USE_SCALAR
	for (int i = 0; i < SIMD_N; i++) {
		v.v[i] = std::copysign(y.v[i], x.v[i]);
	}
#else
	constexpr float signbit = -0.f;
	static simd_float const avx_signbit = simd_float(signbit);
	for( int i = 0; i < SIMD_N; i++) {
		const auto tmp0 = mmx_andnot_ps(avx_signbit.v[i], y.v[i]);
		const auto tmp2 = mmx_and_ps(avx_signbit.v[i], x.v[i]);
		v.v[i] = mmx_or_ps(tmp2, tmp0); // (avx_signbit & from) | (~avx_signbit & to)
	}
#endif
	return v;
}

inline simd_float abs(const simd_float &a) {
	return max(a, -a);
}

inline simd_float max(const simd_float &a, const simd_float &b) {
	simd_float r;
	for (int i = 0; i < SIMD_N; i++) {
		r.v[i] = mmx_max_ps(a.v[i], b.v[i]);
	}
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

#include <cosmictiger/math.hpp>

template<>
class complex<simd_float> {
	simd_float x, y;
public:
	complex<simd_float>() = default;

	complex<simd_float>(simd_float a) {
		x = a;
		y = simd_float(0.0);
	}

	complex<simd_float>(simd_float a, simd_float b) {
		x = a;
		y = b;
	}

	complex<simd_float>& operator*=(simd_float other) {
		x *= other;
		y *= other;
		return *this;
	}

	complex<simd_float>& operator*=(complex<simd_float> other) {
		(*this) = (*this) * other;
		return *this;
	}

	complex<simd_float>& operator+=(complex<simd_float> other) {
		x += other.x;
		y += other.y;
		return *this;
	}

	complex<simd_float>& operator-=(complex<simd_float> other) {
		x -= other.x;
		y -= other.y;
		return *this;
	}

	complex<simd_float> operator*(complex<simd_float> other) const {
		complex<simd_float> a;
		a.x = x * other.x - y * other.y;
		a.y = x * other.y + y * other.x;
		return a;
	}

	complex<simd_float> operator*(cmplx other) const {
		complex<simd_float> a;
		a.x = x * other.x - y * other.y;
		a.y = x * other.y + y * other.x;
		return a;
	}

	complex<simd_float> operator*(float other) const {
		complex<simd_float> a;
		a.x = x * other;
		a.y = x * other;
		return a;
	}

	complex<simd_float> operator/(complex<simd_float> other) const {
		return *this * other.conj() / other.norm();
	}

	complex<simd_float> operator/(simd_float other) const {
		complex<simd_float> b;
		b.x = x / other;
		b.y = y / other;
		return b;
	}

	complex<simd_float> operator*(simd_float other) const {
		complex<simd_float> b;
		b.x = x * other;
		b.y = y * other;
		return b;
	}

	complex<simd_float> operator+(complex<simd_float> other) const {
		complex<simd_float> a;
		a.x = x + other.x;
		a.y = y + other.y;
		return a;
	}

	complex<simd_float> operator-(complex<simd_float> other) const {
		complex<simd_float> a;
		a.x = x - other.x;
		a.y = y - other.y;
		return a;
	}

	complex<simd_float> conj() const {
		complex<simd_float> a;
		a.x = x;
		a.y = -y;
		return a;
	}

	simd_float real() const {
		return x;
	}

	simd_float imag() const {
		return y;
	}

	simd_float& real() {
		return x;
	}

	simd_float& imag() {
		return y;
	}

	simd_float norm() const {
		return ((*this) * conj()).real();
	}

	simd_float abs() const {
		return sqrt(norm());
	}

	complex<simd_float> operator-() const {
		complex<simd_float> a;
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

inline float first(simd_float a) {
	return a[0];
}

CUDA_EXPORT
inline float first(float a) {
	return a;
}

#endif /* COSMICTIGER_SIMD_HPP_ */

