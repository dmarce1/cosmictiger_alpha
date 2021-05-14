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

struct map_workspace_data{
	vector<float> x, y, z, vx, vy, vz;
};

struct map_workspace {
	std::shared_ptr<std::unordered_map<int,map_workspace_data>> data;
};

using map_type = std::shared_ptr<float*>;

int map_add_part(const array<double, NDIM>& Y0, const array<double, NDIM>& Y1,  double tau, double dtau, double dtau_inv, double tau_max,
		map_workspace& ws);
void load_and_save_maps(double tau, double tau_max);
map_workspace get_map_workspace();
void maps_to_file(FILE*fp);
void maps_from_file(FILE*fp);
void cleanup_map_workspace(map_workspace ws, double tau, double dtau, double tau_max);
void healpix2_map(const vector<float>& x, const vector<float>& y, const vector<float>& z,
		const vector<float>& vx, const vector<float>& vy, const vector<float>& vz,  float taui, float tau0, float dtau, map_type map, int Nside);

#endif /* MAP_HPP_ */
