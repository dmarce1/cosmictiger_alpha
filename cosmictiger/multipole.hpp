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
#include <cosmictiger/tensor.hpp>
#include <array>

constexpr int MP = (MORDER*MORDER);

template<class T>
using multipole_type = tensor_trless_sym<T,MORDER>;
using multipole = multipole_type<float>;

struct multi_source {
	multipole m;
	std::array<fixed32, NDIM> x;
};

#endif /* multipole_type_H_ */

