/*
 * multipole_type.hpp
 *
 *  Created on: Jan 30, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_multipole_type_HPP_
#define COSMICTIGER_multipole_type_HPP_

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/array.hpp>
#include <cosmictiger/fixed.hpp>
#include <array>

#include <cosmictiger/spherical_harmonic.hpp>



constexpr int MP = (MORDER*(MORDER+1)/2);

template<class T>
using multipole_type = sphericalY<T,MORDER>;

using multipole = sphericalY<float,MORDER>;

struct multi_source {
	multipole m;
	std::array<fixed32, NDIM> x;
};

#endif /* multipole_type_H_ */

