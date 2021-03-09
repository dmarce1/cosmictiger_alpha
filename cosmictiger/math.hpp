/*
 * math.hpp
 *
 *  Created on: Mar 8, 2021
 *      Author: dmarce1
 */

#ifndef MATH_HPP_
#define MATH_HPP_



#ifdef __CUDA_ARCH__

#define SQRT sqrtf
#define FMA fmaf

#else

#define SQRT sqrt
#define FMA fma

#endif

#endif /* MATH_HPP_ */
