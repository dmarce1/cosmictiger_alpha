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

void denpow_to_phi1(const interp_functor<float> den_k, int N, float box_size);


#endif /* ZELDOVICH_HPP_ */
