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

#ifdef USE_AVX512
#define SIMDALIGN                  __attribute__((aligned(64)))
#define SIMD_VLEN 16
#define _simd_float                 __m512
#define _simd_int                   __m512i
#define _mmx_add_ps(a,b)            _mm512_add_ps((a),(b))
#define _mmx_sub_ps(a,b)            _mm512_sub_ps((a),(b))
#define _mmx_mul_ps(a,b)            _mm512_mul_ps((a),(b))
#define _mmx_div_ps(a,b)            _mm512_div_ps((a),(b))
#define _mmx_add_pd(a,b)            _mm512_add_pd((a),(b))
#define _mmx_sub_pd(a,b)            _mm512_sub_pd((a),(b))
#define _mmx_mul_pd(a,b)            _mm512_mul_pd((a),(b))
#define _mmx_div_pd(a,b)            _mm512_div_pd((a),(b))
#define _mmx_sqrt_ps(a)             _mm512_sqrt_ps(a)
#define _mmx_min_ps(a, b)           _mm512_min_ps((a),(b))
#define _mmx_max_ps(a, b)           _mm512_max_ps((a),(b))
#define _mmx_max_pd(a, b)           _mm512_max_pd((a),(b))
#define _mmx_or_ps(a, b)            _mm512_or_ps((a),(b))
#define _mmx_and_ps(a, b)           _mm512_and_ps((a),(b))
#define _mmx_andnot_ps(a, b)        _mm512_andnot_ps((a),(b))
#define _mmx_rsqrt_ps(a)            _mm512_rsqrt14_ps(a)
#define _mmx_add_epi32(a,b)         _mm512_add_epi32((a),(b))
#define _mmx_sub_epi32(a,b)         _mm512_sub_epi32((a),(b))
#define _mmx_mul_epi32(a,b)         _mm512_mullo_epi32((a),(b))
#define _mmx_cvtps_epi32(a)         _mm512_cvtps_epi32((a))
#define _mmx_fmadd_ps(a,b,c)        _mm512_fmadd_ps ((a),(b),(c))
#define _mmx_cmp_ps(a,b,c)       _mm512_cmp_ps_mask(a,b,c)
#endif

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

#ifdef USE_AVX
#define SIMDALIGN                  __attribute__((aligned(16)))
#define SIMD_VLEN 4
#define _simd_float                 __m128
#define _simd_int                   __m128i
#define _mmx_add_ps(a,b)            _mm_add_ps((a),(b))
#define _mmx_sub_ps(a,b)            _mm_sub_ps((a),(b))
#define _mmx_mul_ps(a,b)            _mm_mul_ps((a),(b))
#define _mmx_div_ps(a,b)            _mm_div_ps((a),(b))
#define _mmx_add_pd(a,b)            _mm_add_pd((a),(b))
#define _mmx_sub_pd(a,b)            _mm_sub_pd((a),(b))
#define _mmx_mul_pd(a,b)            _mm_mul_pd((a),(b))
#define _mmx_div_pd(a,b)            _mm_div_ps((a),(b))
#define _mmx_sqrt_ps(a)             _mm_sqrt_ps(a)
#define _mmx_min_ps(a, b)           _mm_min_ps((a),(b))
#define _mmx_max_ps(a, b)           _mm_max_ps((a),(b))
#define _mmx_max_pd(a, b)           _mm_max_pd((a),(b))
#define _mmx_or_ps(a, b)            _mm_or_ps((a),(b))
#define _mmx_and_ps(a, b)           _mm_and_ps((a),(b))
#define _mmx_andnot_ps(a, b)        _mm_andnot_ps((a),(b))
#define _mmx_rsqrt_ps(a)            _mm_rsqrt_ps(a)
#define _mmx_add_epi32(a,b)         _mm_add_epi32((a),(b))
#define _mmx_sub_epi32(a,b)         _mm_sub_epi32((a),(b))
#define _mmx_mul_epi32(a,b)         _mm_mullo_epi32((a),(b))
#define _mmx_cvtps_epi32(a)         _mm_cvtps_epi32((a))
#define _mmx_cmp_ps(a,b,c)        	_mm_cmp_ps(a,b,c)
#endif

class simd_float;
class simd_int;

class simd_float {
private:
	_simd_float v[2];
public:
	static constexpr std::size_t size() {
		return 2 * SIMD_VLEN;
	}
	simd_float() = default;
	inline ~simd_float() = default;
	simd_float(const simd_float&) = default;
	inline simd_float(float d) {
#ifdef USE_AVX512
        v[0] = _mm512_set_ps(d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d);
        v[1] = _mm512_set_ps(d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d);
#endif
#ifdef USE_AVX2
		v[0] = _mm256_set_ps(d, d, d, d, d, d, d, d);
		v[1] = _mm256_set_ps(d, d, d, d, d, d, d, d);
#endif
#ifdef USE_AVX
		v[0] = _mm_set_ps(d, d, d, d);
		v[1] = _mm_set_ps(d, d, d, d);
#endif
	}
	inline float sum() const {
		simd_float r = *this;
		for (int N = simd_float::size() / 4; N > 0; N >>= 1) {
			r.v[0] = _mmx_add_ps(r.v[0], r.v[1]);
			for (int i = 0; i < N; i++) {
				r.v[1][i] = r.v[0][i + N];
			}
		}
		return r.v[0][0] + r.v[0][1];
	}

	inline simd_float(simd_int i);

	inline simd_float& operator=(const simd_float &other) = default;
	simd_float& operator=(simd_float &&other) = default;
	inline simd_float operator+(const simd_float &other) const {
		simd_float r;
		r.v[0] = _mmx_add_ps(v[0], other.v[0]);
		r.v[1] = _mmx_add_ps(v[1], other.v[1]);
		return r;
	}
	inline simd_float operator-(const simd_float &other) const {
		simd_float r;
		r.v[0] = _mmx_sub_ps(v[0], other.v[0]);
		r.v[1] = _mmx_sub_ps(v[1], other.v[1]);
		return r;
	}
	inline simd_float operator*(const simd_float &other) const {
		simd_float r;
		r.v[0] = _mmx_mul_ps(v[0], other.v[0]);
		r.v[1] = _mmx_mul_ps(v[1], other.v[1]);
		return r;
	}
	inline simd_float operator/(const simd_float &other) const {
		simd_float r;
		r.v[0] = _mmx_div_ps(v[0], other.v[0]);
		r.v[1] = _mmx_div_ps(v[1], other.v[1]);
		return r;
	}
	inline simd_float operator+() const {
		return *this;
	}
	inline simd_float operator-() const {
		return simd_float(0.0) - *this;
	}
	inline simd_float& operator+=(const simd_float &other) {
		v[0] = _mmx_add_ps(v[0], other.v[0]);
		v[1] = _mmx_add_ps(v[1], other.v[1]);
		return *this;
	}
	inline simd_float& operator-=(const simd_float &other) {
		v[0] = _mmx_sub_ps(v[0], other.v[0]);
		v[1] = _mmx_sub_ps(v[1], other.v[1]);
		return *this;
	}
	inline simd_float& operator*=(const simd_float &other) {
		v[0] = _mmx_mul_ps(v[0], other.v[0]);
		v[1] = _mmx_mul_ps(v[1], other.v[1]);
		return *this;
	}
	inline simd_float& operator/=(const simd_float &other) {
		v[0] = _mmx_div_ps(v[0], other.v[0]);
		v[1] = _mmx_div_ps(v[1], other.v[1]);
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
		return reinterpret_cast<float*>(&v)[i];
	}
	inline float operator[](std::size_t i) const {
		return reinterpret_cast<const float*>(&v)[i];
	}

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
		auto mask0 = _mmx_cmp_ps(v[0], other.v[0], _CMP_LT_OQ);
		auto mask1 = _mmx_cmp_ps(v[1], other.v[1], _CMP_LT_OQ);
#ifdef USE_AVX512
		rc.v[0] = _mm512_mask_mov_ps(zero.v[0],mask0,one.v[0]);
		rc.v[1] = _mm512_mask_mov_ps(zero.v[1],mask1,one.v[1]);
#else
		rc.v[0] = _mmx_and_ps(mask0, one.v[0]);
		rc.v[1] = _mmx_and_ps(mask1, one.v[1]);
#endif
		return rc;
	}
	simd_float operator>(simd_float other) const { // 2
		simd_float rc;
		static const simd_float one(1);
		static const simd_float zero(0);
		auto mask0 = _mmx_cmp_ps(v[0], other.v[0], _CMP_GT_OQ);
		auto mask1 = _mmx_cmp_ps(v[1], other.v[1], _CMP_GT_OQ);
#ifdef USE_AVX512
		rc.v[0] = _mm512_mask_mov_ps(zero.v[0],mask0,one.v[0]);
		rc.v[1] = _mm512_mask_mov_ps(zero.v[1],mask1,one.v[1]);
#else
		rc.v[0] = _mmx_and_ps(mask0, one.v[0]);
		rc.v[1] = _mmx_and_ps(mask1, one.v[1]);
#endif
		return rc;
	}
}SIMDALIGN;
;

class simd_int {
private:
	_simd_int v[2];
public:
	static constexpr std::size_t size() {
		return 2 * SIMD_VLEN;
	}
	simd_int() = default;
	inline ~simd_int() = default;
	simd_int(const simd_int&) = default;
	inline simd_int(int d) {
#ifdef USE_AVX512
        v[0] = _mm512_set_epi32(d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d);
        v[1] = _mm512_set_epi32(d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d);
#endif
#ifdef USE_AVX2
		v[0] = _mm256_set_epi32(d, d, d, d, d, d, d, d);
		v[1] = _mm256_set_epi32(d, d, d, d, d, d, d, d);
#endif
#ifdef USE_AVX
		v[0] = _mm_set_epi32(d, d, d, d);
		v[1] = _mm_set_epi32(d, d, d, d);
#endif
	}
	inline simd_int& operator=(const simd_int &other) = default;
	simd_int& operator=(simd_int &&other) = default;
	inline simd_int operator+(const simd_int &other) const {
		simd_int r;
		r.v[0] = _mmx_add_epi32(v[0], other.v[0]);
		r.v[1] = _mmx_add_epi32(v[1], other.v[1]);
		return r;
	}
	inline simd_int operator-(const simd_int &other) const {
		simd_int r;
		r.v[0] = _mmx_sub_epi32(v[0], other.v[0]);
		r.v[1] = _mmx_sub_epi32(v[1], other.v[1]);
		return r;
	}
	inline simd_int operator*(const simd_int &other) const {
		simd_int r;
		r.v[0] = _mmx_mul_epi32(v[0], other.v[0]);
		r.v[1] = _mmx_mul_epi32(v[1], other.v[1]);
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
		return reinterpret_cast<int*>(&v)[i];
	}
	inline int operator[](std::size_t i) const {
		return reinterpret_cast<const int*>(&v)[i];
	}
	friend class simd_float;

}SIMDALIGN;

inline simd_float::simd_float(simd_int i) {
#ifdef USE_AVX512
	v[0] = _mm512_cvtepi32_ps(i.v[0]);
	v[1] = _mm512_cvtepi32_ps(i.v[1]);
#endif
#ifdef USE_AVX2
	v[0] = _mm256_cvtepi32_ps(i.v[0]);
	v[1] = _mm256_cvtepi32_ps(i.v[1]);
#endif

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
#ifdef USE_AVX512
    __m512i n[2];
#endif
#ifdef USE_AVX2
	__m256i n[2];
#endif
#ifdef USE_AVX512
	r0.v[0] = _mm512_roundscale_ps(r.v[0], _MM_FROUND_TO_NEAREST_INT);
	r0.v[1] = _mm512_roundscale_ps(r.v[1], _MM_FROUND_TO_NEAREST_INT);
	n[0] = _mm512_cvtps_epi32(r0.v[0]);
	n[1] = _mm512_cvtps_epi32(r0.v[1]);
#endif
#ifdef USE_AVX2
	r0.v[0] = _mm256_round_ps(r.v[0], _MM_FROUND_TO_NEAREST_INT);							// 1
	r0.v[1] = _mm256_round_ps(r.v[1], _MM_FROUND_TO_NEAREST_INT);							// 1
	n[0] = _mm256_cvtps_epi32(r0.v[0]);														// 1
	n[1] = _mm256_cvtps_epi32(r0.v[1]);														// 1
#endif
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
#ifdef USE_AVX512
	static const auto sevenf =  _mm512_set_epi32(0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f);
	auto imm00 = _mm512_add_epi32(n[0], sevenf); // 1
	auto imm01 = _mm512_add_epi32(n[1], sevenf); // 1
	imm00 = _mm512_slli_epi32(imm00, 23);
	imm01 = _mm512_slli_epi32(imm01, 23);
	r0.v[0] = _mm512_castsi512_ps(imm00);
	r0.v[1] = _mm512_castsi512_ps(imm01);
#endif
#ifdef USE_AVX2
	static const auto sevenf = _mm256_set_epi32(0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f);
	auto imm00 = _mm256_add_epi32(n[0], sevenf);
	auto imm01 = _mm256_add_epi32(n[1], sevenf);
	imm00 = _mm256_slli_epi32(imm00, 23);
	imm01 = _mm256_slli_epi32(imm01, 23);
	r0.v[0] = _mm256_castsi256_ps(imm00);
	r0.v[1] = _mm256_castsi256_ps(imm01);
#endif
	auto res = y * r0;																			// 1
	return res;
}

inline simd_float round(const simd_float a) {
	simd_float v;
#ifdef USE_AVX512
	v.v[0] = _mm512_roundscale_ps(a.v[0], 0);
	v.v[1] = _mm512_roundscale_ps(a.v[1], 0);
#endif
#ifdef USE_AVX2
	v.v[0] = _mm256_round_ps(a.v[0], _MM_FROUND_TO_NEAREST_INT);
	v.v[1] = _mm256_round_ps(a.v[1], _MM_FROUND_TO_NEAREST_INT);
#endif
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
	v.v[0] = _mmx_fmadd_ps(a.v[0], b.v[0], c.v[0]);
	v.v[1] = _mmx_fmadd_ps(a.v[1], b.v[1], c.v[1]);
	return v;
}

inline simd_float sqrt(const simd_float &vec) {
	simd_float r;
	r.v[0] = _mmx_sqrt_ps(vec.v[0]);
	r.v[1] = _mmx_sqrt_ps(vec.v[1]);
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
	r.v[0] = _mmx_min_ps(a.v[0], b.v[0]);
	r.v[1] = _mmx_min_ps(a.v[1], b.v[1]);
	return r;
}

inline simd_float max(const simd_float &a, const simd_float &b) {
	simd_float r;
	r.v[0] = _mmx_max_ps(a.v[0], b.v[0]);
	r.v[1] = _mmx_max_ps(a.v[1], b.v[1]);
	return r;
}

#endif /* COSMICTIGER_SIMD_HPP_ */
