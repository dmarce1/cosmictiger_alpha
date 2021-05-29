/*
 * zeldovich.hpp
 *
 *  Created on: Jan 12, 2021
 *      Author: dmarce1
 */

#ifndef ZELDOVICH_HPP_
#define ZELDOVICH_HPP_

#include <cosmictiger/vector.hpp>
#include <cosmictiger/interp.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/fourier.hpp>

enum zeldovich_t {
	DENSITY, DISPLACEMENT, VELOCITY
};

__global__ void zeldovich(cmplx* den, const cmplx* rands, const interp_functor<float>* P, float box_size, int N,
		int dim, zeldovich_t);
__global__
void _2lpt_kernel(cmplx* Y, int xbegin, int xend, const interp_functor<float> den_k, int N, float box_size, int dim1,
		int dim2);
void execute_2lpt_kernel(cmplx* Y, int xbegin, int xend, const interp_functor<float> den_k, int N, float box_size, int dim1,
		int dim2);

void _2lpt(const interp_functor<float> den_k, int N, float box_size, int = NDIM, int = NDIM, int seed = 42);
float phi1_to_particles(int N, float box_size, float D1, float prefactor, int);


#endif /* ZELDOVICH_HPP_ */
