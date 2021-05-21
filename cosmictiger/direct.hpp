/*
 * direct.hpp
 *
 *  Created on: May 20, 2021
 *      Author: dmarce1
 */

#ifndef DIRECT_HPP_
#define DIRECT_HPP_


#include <cosmictiger/defs.hpp>
#include <cosmictiger/array.hpp>
#include <cosmictiger/fixed.hpp>

struct gforce {
	array<float, NDIM> f;
	float phi;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & f;
		arc & phi;
	}
};

std::vector<gforce> cuda_direct(std::vector<std::array<fixed32, NDIM>> pts);

void direct_force_test();

#endif /* DIRECT_HPP_ */
