/*
 * power.hpp
 *
 *  Created on: Apr 11, 2021
 *      Author: dmarce1
 */

#ifndef POWER_HPP_
#define POWER_HPP_

#include <cosmictiger/defs.hpp>
#include <cosmictiger/cuda.hpp>

CUDA_EXPORT
float eisenstein_and_hu(float k, float omega_c, float omega_b, float hubble);

#endif /* POWER_HPP_ */
