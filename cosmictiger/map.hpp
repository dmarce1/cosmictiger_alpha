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
#include <cosmictiger/vector.hpp>
#include <unordered_map>
#include <memory>

using map_workspace = std::shared_ptr<std::unordered_map<int,std::array<vector<float>,NDIM>>>;

using map_type = std::shared_ptr<vector<float>>;

int map_add_part(const array<double, NDIM>& Y0, const array<double, NDIM>& Y1, double tau, double dtau, double tau_max, map_workspace& ws);
void load_and_save_maps(double tau, double tau_max);
map_workspace get_map_workspace();
void cleanup_map_workspace(map_workspace ws);
void healpix2_map(const vector<float>& x, const vector<float>& y, const vector<float>& z, map_type map, int Nside);

#endif /* MAP_HPP_ */
