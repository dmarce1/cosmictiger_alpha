/*
 * rockstar.hpp
 *
 *  Created on: Jun 21, 2021
 *      Author: dmarce1
 */

#ifndef ROCKSTAR_HPP_
#define ROCKSTAR_HPP_

#include <cosmictiger/defs.hpp>
#include <cosmictiger/array.hpp>

struct halo_part {
	array<float, NDIM> x;
	float phi;
};

inline void swap(halo_part& a, halo_part& b) {
	const auto tmp = a.x;
	a.x = b.x;
	b.x = tmp;
}

struct halo_tree {
	array<int, NCHILD> children;
	array<float, NDIM> x;
	float mass;
	float radius;
};

#endif /* ROCKSTAR_HPP_ */
