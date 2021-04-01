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



struct multipole_pos {
	multipole multi;
	array<fixed32, NDIM> pos;
};


struct multi_crit {
	array<fixed32, NDIM> pos;
	float r;
};


using children_type = array<tree_ptr,NCHILD>;
using parts_type = pair<size_t,size_t>;

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

	CUDA_EXPORT
	inline multipole_pos& multi_pos(size_t i) {
		return multi_data_[i];
	}

	CUDA_EXPORT
	inline const multipole_pos& multi_pos(size_t i) const {
		return multi_data_[i];
	}

	CUDA_EXPORT
	inline children_type& children(size_t i) {
		return child_data_[i];
	}

	CUDA_EXPORT
	inline const children_type& children(size_t i) const {
		return child_data_[i];
	}

	CUDA_EXPORT
	inline parts_type& parts(size_t i) {
		return parts_data_[i];
	}

	CUDA_EXPORT
	inline const parts_type& parts(size_t i) const {
		return parts_data_[i];
	}

	CUDA_EXPORT
	inline multi_crit& mcrit(size_t i) {
		return crit_data_[i];
	}

	CUDA_EXPORT
	inline const multi_crit& mcrit(size_t i) const {
		return crit_data_[i];
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
