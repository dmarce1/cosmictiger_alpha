/*
 * ewald_indices.hpp
 *
 *  Created on: Mar 8, 2021
 *      Author: dmarce1
 */

#ifndef EWALD_INDICES_HPP_
#define EWALD_INDICES_HPP_

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/array.hpp>
#include <cosmictiger/expansion.hpp>

struct ewald_data {
	CUDA_EXPORT static int nreal();
	CUDA_EXPORT static int nfour();
	CUDA_EXPORT static const float* real_index(int);
	CUDA_EXPORT static const float* four_index(int);
	CUDA_EXPORT static const float* periodic_part(int);
};

#endif /* EWALD_INDICES_HPP_ */
