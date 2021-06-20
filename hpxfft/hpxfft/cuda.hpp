/*
 * cuda.hpp
 *
 *  Created on: Jun 20, 2021
 *      Author: dmarce1
 */

#ifndef CUDA_HPP_
#define CUDA_HPP_

#include <hpxfft/print.hpp>

#define CUDA_CHECK( a ) if( a != cudaSuccess ) PRINT( "CUDA error on line %i of %s : %s\n", __LINE__, __FILE__, cudaGetErrorString(a))


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

#ifdef __CUDACC__
#define CUDA_EXPORT __device__ __host__
#else
#define CUDA_EXPORT
#endif


void cuda_set_device();

#endif /* CUDA_HPP_ */
