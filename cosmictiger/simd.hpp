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
