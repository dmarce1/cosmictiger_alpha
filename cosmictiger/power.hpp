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
#include <cosmictiger/particle.hpp>
#include <cosmictiger/math.hpp>

void compute_particle_power_spectrum(particle_set& parts, int filenum);
void compute_power_spectrum(cmplx* den, float* spec, int N);
__global__ void power_spectrum_init(particle_set partsets, cmplx* den_k, size_t N, float mass0);

#endif /* POWER_HPP_ */
