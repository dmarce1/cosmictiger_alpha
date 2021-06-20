/*
 * rand.hpp
 *
 *  Created on: Jan 25, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_RAND_HPP_
#define COSMICTIGER_RAND_HPP_

#include <cosmictiger/fixed.hpp>

fixed32 rand_fixed32();


__global__
void generate_random_vectors(fixed32* x, fixed32* y, fixed32* z, size_t N, int seed);

#endif /* COSMICTIGER_RAND_HPP_ */
