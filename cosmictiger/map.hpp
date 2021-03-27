/*
 * map.hpp
 *
 *  Created on: Mar 25, 2021
 *      Author: dmarce1
 */

#ifndef MAP_HPP_
#define MAP_HPP_


#include <cosmictiger/defs.hpp>
#include <cosmictiger/array.hpp>

int map_add_part(const array<double, NDIM>& Y0, const array<double, NDIM>& Y1, double tau, double dtau, double tau_max);
void load_and_save_maps(double tau, double tau_max);


#endif /* MAP_HPP_ */
