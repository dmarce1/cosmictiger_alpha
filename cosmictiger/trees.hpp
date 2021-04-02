/*
 * trees.hpp
 *
 *  Created on: Apr 1, 2021
 *      Author: dmarce1
 */

#ifndef TREES_HPP_
#define TREES_HPP_

#include <cosmictiger/defs.hpp>
#include <cosmictiger/memory.hpp>
#include <cosmictiger/array.hpp>
#include <cosmictiger/fixed.hpp>
#include <cosmictiger/multipole.hpp>
#include <cosmictiger/tree_ptr.hpp>

#include <atomic>

inline int ewald_min_level(double theta, double h) {
	int lev = 12;
	while (1) {
		int N = 1 << (lev / NDIM);
		double dx = EWALD_MIN_DIST * N;
		double a;
		constexpr double ffac = 1.01;
		if (lev % NDIM == 0) {
			a = std::sqrt(3) + ffac * h;
		} else if (lev % NDIM == 1) {
			a = 1.5 + ffac * h;
		} else {
			a = std::sqrt(1.5) + ffac * h;
		}
		double r = (1.0 + SINK_BIAS) * a / theta + h * N;
		if (dx > r) {
			break;
		}
		lev++;
	}
	return lev;
}

class trees {
private:
	bool initialized;
	multipole_pos* multi_data_;
	children_type* child_data_;
	parts_type* parts_data_;
	multi_crit* crit_data_;
	int ntrees_;
	int nchunks_;
	int chunk_size_;
public:
#ifndef __CUDACC__
	trees() {
		initialized = false;
	}
#endif
	int get_chunk();
	int chunk_size() const;
	void init();
	void clear();

	inline int get_tree_count() const {
		return ntrees_;
	}

	CUDA_EXPORT
	inline const multipole_pos& get_multi(int i) const {
		assert(i >= 0);
		assert(i < ntrees_);
		return multi_data_[i];
	}

	CUDA_EXPORT
	inline
	void set_multi(int i, const multipole& mpole, const array<fixed32, NDIM>& pos) {
		assert(i >= 0);
		assert(i < ntrees_);
		multi_data_[i].pos = pos;
		multi_data_[i].multi = mpole;
	}

	CUDA_EXPORT
	inline children_type get_children(int i) const {
		assert(i >= 0);
		assert(i < ntrees_);
		return child_data_[i];
	}

	CUDA_EXPORT
	inline
	void set_children(int i, const children_type& c) {
		assert(i >= 0);
		assert(i < ntrees_);
		child_data_[i] = c;
	}

	CUDA_EXPORT
	inline parts_type get_parts(int i) const {
		assert(i >= 0);
		assert(i < ntrees_);
		return parts_data_[i];
	}

	CUDA_EXPORT
	inline
	void set_parts(int i, parts_type p) {
		assert(i >= 0);
		assert(i < ntrees_);
		parts_data_[i] = p;
	}

	CUDA_EXPORT
	inline multi_crit get_mcrit(int i) const {
		assert(i >= 0);
		assert(i < ntrees_);
		return crit_data_[i];
	}

	CUDA_EXPORT
	inline
	void set_mcrit(int i, const array<fixed32, NDIM>& pos, float r) {
		assert(i >= 0);
		assert(i < ntrees_);
		crit_data_[i].pos = pos;
		crit_data_[i].r = r;
	}

};

class trees_allocator {
	int alloc;
	int max_alloc;
	static trees tree_database;
public:
	static void init() {
		tree_database.init();
	}
	inline trees_allocator() {
		alloc = tree_database.get_chunk();
		max_alloc = alloc + tree_database.chunk_size();
	}
	inline void cleanup() {
		tree_database.clear();
	}
	static inline trees& get_trees() {
		return tree_database;
	}
	inline tree_ptr get_tree_ptr() {
		int index = alloc++;
		if (index == 0) {
			index = alloc++;
		}
		if (alloc == max_alloc) {
			alloc = tree_database.get_chunk();
			max_alloc = alloc + tree_database.chunk_size();
		}
		tree_ptr ptr;
		ptr.index = index;
		assert(ptr.index>=0);
		assert(ptr.index < tree_database.get_tree_count());
		return ptr;
	}
};

#ifdef __CUDA_ARCH__
__device__ trees& get_trees_database();
#endif

#ifndef __CUDA_ARCH__
trees& get_trees_database();
#endif

trees& get_cuda_trees_database();

#endif /* TREES_HPP_ */
