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

__global__
void _2lpt_kernel(cmplx* Y, int xbegin,const interp_functor<float> den_k, int N, float box_size, int dim1,
		int dim2);
void execute_2lpt_kernel(std::vector<cmplx>& Y, int xbegin, int xend, const interp_functor<float> den_k, int N, float box_size, int dim1,
		int dim2);

__global__
void _2lpt_correction_kernel(cmplx* Y, int xbegin, int xend, int N, float box_size, int dim);
void execute_2lpt_correction_kernel( std::vector<cmplx>& Y, int xbegin, int xend, int N, float box_size, int dim);

void _2lpt(const interp_functor<float> den_k, int N, float box_size, int = NDIM, int = NDIM, int seed = 42);
void _2lpt_correction1(int N, float );
void _2lpt_correction2(int N, float, int );
float phi1_to_particles(int N, float box_size, float D1, float prefactor, int);
float phi2_to_particles(int N, float box_size, float D1, float prefactor, int);

void _2lpt_destroy();
void _2lpt_init(int N);
void _2lpt_phase(int N, int phase);

#endif /* ZELDOVICH_HPP_ */
