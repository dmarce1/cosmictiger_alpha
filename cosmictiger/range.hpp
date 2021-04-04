#pragma once

#include <cosmictiger/defs.hpp>
#include <cosmictiger/fixed.hpp>

#include <array>

#define NCORNERS (1<<NDIM)

struct range {
	std::array<double, NDIM> begin;
	std::array<double, NDIM> end;

	inline bool contains(std::array<fixed32, NDIM> v) const {
		bool rc = true;
		for (int dim = 0; dim < NDIM; dim++) {
			if (v[dim].to_double() < begin[dim] || v[dim].to_double() > end[dim]) {
				rc = false;
				break;
			}
		}
		return rc;
	}
	template<class A>
	inline void serialize(A&& arc, unsigned) {
		arc & begin;
		arc & end;
	}
	bool intersects(const range& other) const {
		bool rc = true;
		for (int dim = 0; dim < NDIM; dim++) {
			if (end[dim] < other.begin[dim]) {
				rc = false;
				break;
			}
			if (begin[dim] > other.end[dim]) {
				rc = false;
				break;
			}
		}
		return rc;
	}
	inline std::array<std::array<fixed64, NDIM>, NCORNERS> get_corners() {
		std::array<std::array<fixed64, NDIM>, NCORNERS> v;
		for (int ci = 0; ci < NCORNERS; ci++) {
			for (int dim = 0; dim < NDIM; dim++) {
				if ((ci >> 1) & 1) {
					v[ci][dim] = begin[dim];
				} else {
					v[ci][dim] = end[dim];
				}
			}
		}
		return v;
	}

};
