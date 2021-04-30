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
#include <cosmictiger/particle_sets.hpp>
#include <cosmictiger/math.hpp>

void compute_particle_power_spectrum(particle_sets& parts, int filenum);
void compute_power_spectrum(cmplx* den, float* spec, int N);
__global__ void power_spectrum_init(particle_sets partsets, cmplx* den_k, size_t N, float mass0, bool sph);
void eisenstein_and_hu(std::function<float(float)>& Pc, std::function<float(float)>& Pb, float omega_b, float omega_c,
		float Theta, float h, float ns);

#endif /* POWER_HPP_ */
