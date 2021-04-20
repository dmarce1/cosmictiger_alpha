#pragma once

#include <cosmictiger/defs.hpp>
#include <cosmictiger/fixed.hpp>
#include <cosmictiger/array.hpp>

#define NCORNERS (1<<NDIM)

struct range {
	array<double, NDIM> begin;
	array<double, NDIM> end;

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
	CUDA_EXPORT
	inline void expand(double len) {
		for (int dim = 0; dim < NDIM; dim++) {
			begin[dim] -= len;
			end[dim] += len;
		}
	}
	CUDA_EXPORT
	inline bool intersects(const range& other) const {
		bool rc = true;
#ifndef __CUDA_ARCH__
		using namespace std;
#endif
		for (int dim = 0; dim < NDIM; dim++) {
			bool this_rc;
			if (min(end[dim], other.end[dim]) - max(begin[dim], other.begin[dim]) >= 0.0) {
				this_rc = true;
			} else if (min(end[dim] + 1.0, other.end[dim]) - max(begin[dim] + 1.0, other.begin[dim]) >= 0.0) {
				this_rc = true;
			} else if (min(end[dim] - 1.0, other.end[dim]) - max(begin[dim] - 1.0, other.begin[dim]) >= 0.0) {
				this_rc = true;
			} else {
				this_rc = false;
			}
			if (!this_rc) {
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
