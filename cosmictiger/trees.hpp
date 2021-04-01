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


class trees {
private:
	void* data_;
	multipole_pos* multi_data_;
	children_type* child_data_;
	parts_type* parts_data_;
	multi_crit* crit_data_;
	size_t ntrees_;
	size_t nchunks_;
	size_t chunk_size_;
	std::atomic<size_t> next_chunk_;
public:
	size_t get_chunk();
	size_t chunk_size() const;
	trees();
	~trees();
	void clear();

	CUDA_EXPORT inline
	multipole_pos get_multi(size_t i) const {
		return multi_data_[i];
	}

	CUDA_EXPORT inline
	void set_multi(size_t i, const multipole& mpole, const array<fixed32,NDIM>& pos ) {
		multi_data_[i].pos = pos;
		multi_data_[i].multi = mpole;
	}

	CUDA_EXPORT inline
	children_type get_children(size_t i) const {
		return child_data_[i];
	}

	CUDA_EXPORT inline
	void set_children(size_t i, const children_type& c) {
		child_data_[i] = c;
	}

	CUDA_EXPORT inline
	parts_type get_parts(size_t i) const {
		return parts_data_[i];
	}

	CUDA_EXPORT inline
	void set_parts(size_t i, parts_type p) {
		parts_data_[i] = p;
	}

	CUDA_EXPORT inline
	multi_crit get_mcrit(size_t i) const {
		return crit_data_[i];
	}

	CUDA_EXPORT inline
	void set_mcrit(size_t i, const array<fixed32,NDIM>& pos, float r ) {
		crit_data_[i].pos = pos;
		crit_data_[i].r = r;
	}

};

class trees_allocator {
	trees& nodes;
	size_t alloc;
	size_t max_alloc;
public:
	inline trees_allocator(trees& nodes_) :
			nodes(nodes_) {
		alloc = nodes.get_chunk();
		max_alloc = alloc + nodes.chunk_size();
	}
	inline size_t get_tree_index() {
		size_t index = alloc++;
		if (alloc == max_alloc) {
			alloc = nodes.get_chunk();
			max_alloc = alloc + nodes.chunk_size();
		}
		return index;
	}
};

#endif /* TREES_HPP_ */
