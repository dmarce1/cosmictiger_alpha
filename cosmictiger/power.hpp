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

void matter_power_spectrum_init();
void matter_power_spectrum(int filenum);


#endif /* POWER_HPP_ */
