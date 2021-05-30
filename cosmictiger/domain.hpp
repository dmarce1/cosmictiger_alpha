/*
 * domain.hpp
 *
 *  Created on: May 17, 2021
 *      Author: dmarce1
 */

#ifndef DOMAIN_HPP_
#define DOMAIN_HPP_

#include <cosmictiger/range.hpp>
#include <cosmictiger/memory.hpp>
#include <cosmictiger/hpx.hpp>

struct domain_bounds {
	fixed32 bound;
	int xdim;
	domain_bounds* left;
	domain_bounds* right;
	void create_uniform_bounds(range box, int, int);
	void create_uniform_bounds();
	void destroy();

	domain_bounds() {
		left = right = nullptr;
	}

	CUDA_EXPORT
	int find_proc(const array<fixed32, NDIM>&, int nbegin = -1, int nend = -1) const;

	CUDA_EXPORT
	range find_proc_range(int proc, int nbegin, int nend, range box) const;

	CUDA_EXPORT
	range find_proc_range(int proc) const;

	CUDA_EXPORT
	range find_range(pair<int, int>, int begin, int end, range box) const;

	CUDA_EXPORT
	range find_range(pair<int, int>) const;

#ifndef __CUDACC__
	HPX_SERIALIZATION_SPLIT_MEMBER()
	;

	template<class A>
	void save(A&& arc, unsigned) const {
		bool leaf = left == nullptr;
		arc & leaf;
		arc & bound;
		arc & xdim;
		if (!leaf) {
			arc & *left;
			arc & *right;
		}
	}

	template<class A>
	void load(A&& arc, unsigned) {
		bool leaf;
		arc & leaf;
		arc & bound;
		arc & xdim;
		if (!leaf) {
			unified_allocator alloc;
			CUDA_MALLOC(left,1);
			CUDA_MALLOC(right,1);
			arc & *left;
			arc & *right;
		}
	}
#endif
};

#endif /* DOMAIN_HPP_ */
