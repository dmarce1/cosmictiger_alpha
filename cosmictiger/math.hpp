#pragma once

#include <cosmictiger/defs.hpp>

#define FMAX max
#define FMIN min
#define EXP exp
#define RSQRT rsqrt
#define SQRT sqrt
#define ABS fabs
#define SINCOS sincos
#define FMA fma

template<class T>
inline T sqr(T a) {
	return a * a;
}

template<class T>
inline T sqr(T a, T b, T c) {
	return fmaf(a, a, fmaf(b, b, sqr(c)));
}


/*
 * math.hpp
 *
 *  Created on: Jan 10, 2021
 *      Author: dmarce1
 */

#ifndef GPUTIGER_MATH_HPP_
#define GPUTIGER_MATH_HPP_

#include <cstdio>
#include <cstdint>

#include <functional>

/*#define POW(a,b) powf(a,b)
 #define LOG(a) logf(a)
 #define EXP(a) expf(a)
 #define SQRT(a) sqrtf(a)*/
#define COS(a) cosf(a)
#define SIN(a) sinf(a)
//#define SINCOS(a,b,c) sincosf(a,b,c)

double find_root(std::function<double(double)> f);

template<class T>
 inline T round_up(T num, T mod) {
	return ((num - 1) / mod + 1) * mod;
}

#endif /* GPUTIGER_MATH_HPP_ */

