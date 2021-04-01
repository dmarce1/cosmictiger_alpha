/*
 * tree_ptr.hpp
 *
 *  Created on: Apr 1, 2021
 *      Author: dmarce1
 */

#ifndef TREE_PTR_HPP_
#define TREE_PTR_HPP_


#include <cosmictiger/fast_future.hpp>

struct tree;
struct kick_params_type;



struct multipole_pos {
	multipole multi;
	array<fixed32, NDIM> pos;
};


struct multi_crit {
	array<fixed32, NDIM> pos;
	float r;
};

struct tree_ptr;

using children_type = array<tree_ptr,NCHILD>;

using parts_type = pair<size_t,size_t>;

struct active_type {
	size_t nodes;
	size_t parts;
};

class tree_ptr {
	uintptr_t ptr;

public:

	inline tree_ptr& operator=(tree* ptr_) {
		ptr = (uintptr_t) (ptr_);
		return *this;
	}

	CUDA_EXPORT
	inline multi_crit get_mcrit() const;

	CUDA_EXPORT
	inline const multipole_pos& get_mpole() const;

	CUDA_EXPORT
	inline parts_type get_parts() const;

	CUDA_EXPORT
	inline children_type get_children() const;

	CUDA_EXPORT
	inline active_type get_active() const;

	CUDA_EXPORT
	bool is_leaf() const;
#ifndef __CUDACC__
	hpx::future<void> kick(kick_params_type*, bool);
#endif

	friend class tree;
};




#endif /* TREE_PTR_HPP_ */
