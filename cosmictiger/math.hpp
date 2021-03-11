#pragma once

#include <cosmictiger/defs.hpp>

#ifdef __CUDA_ARCH__
#define FMAX fmaxf
#define FMIN fminf
#define EXP expf
#define RSQRT rsqrtf
#define SQRT sqrtf
#define ABS fabsf
#define SINCOS sincosf
#define FMA fmaf
#else
#define FMAX max
#define FMIN min
#define EXP exp
#define RSQRT rsqrt
#define SQRT sqrt
#define ABS fabs
#define SINCOS sincos
#define FMA fma
#endif


template<class T>
CUDA_EXPORT inline T sqr(T a) {
   return a * a;
}


CUDA_DEVICE inline float erfcexp(const float &x, float *e) {				// 18 + FLOP_DIV + FLOP_EXP
	const float p(0.3275911f);
	const float a1(0.254829592f);
	const float a2(-0.284496736f);
	const float a3(1.421413741f);
	const float a4(-1.453152027f);
	const float a5(1.061405429f);
	const float t1 = 1.f / fma(p, x, 1.f);			            // FLOP_DIV + 2
	const float t2 = t1 * t1;											// 1
	const float t3 = t2 * t1;											// 1
	const float t4 = t2 * t2;											// 1
	const float t5 = t2 * t3;											// 1
	*e = EXP(-x * x);												  // 2 + FLOP_EXP
	return FMA(a1, t1, FMA(a2, t2, FMA(a3, t3, FMA(a4, t4, a5 * t5)))) * *e; 			// 10
}


