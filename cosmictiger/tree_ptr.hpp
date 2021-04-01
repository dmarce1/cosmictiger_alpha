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

struct tree_ptr {
	uintptr_t ptr;
//   int rank;
//	int8_t opened;
#ifndef NDEBUG
	int constructed;
#endif
#ifndef NDEBUG
	CUDA_EXPORT
	inline tree_ptr() {
		//   rank = -1;
		ptr = 0;
//		opened = false;
		constructed = 1234;
	}
#else
	tree_ptr() = default;
#endif
	CUDA_EXPORT
	inline tree_ptr(tree_ptr &&other) {
		//  rank = other.rank;
		ptr = other.ptr;
//		opened = other.opened;
#ifndef NDEBUG
		constructed = 1234;
#endif
	}
	CUDA_EXPORT
	inline tree_ptr(const tree_ptr &other) {
		//  rank = other.rank;
		ptr = other.ptr;
//		opened = other.opened;
#ifndef NDEBUG
		constructed = 1234;
#endif
	}
	CUDA_EXPORT
	inline tree_ptr& operator=(const tree_ptr &other) {
		assert(constructed == 1234);
		ptr = other.ptr;
		// rank = other.rank;
//		opened = other.opened;
		return *this;
	}
	CUDA_EXPORT
	inline tree_ptr& operator=(tree_ptr &&other) {
		assert(constructed == 1234);
		ptr = other.ptr;
		// rank = other.rank;
//		opened = other.opened;
		return *this;
	}
	CUDA_EXPORT
	inline bool operator==(const tree_ptr &other) const {
		assert(constructed == 1234);
		return /*rank == other.rank && */ptr == other.ptr/* && opened == other.opened*/;
	}
	template<class A>
	void serialization(A &&arc, unsigned) {
		arc & ptr;
		//   arc & rank;
//		arc & opened;
	}
	CUDA_EXPORT

	inline operator tree*() {
		assert(constructed == 1234);
		assert(ptr);
		return (tree*) (ptr);
	}
	CUDA_EXPORT
	inline operator const tree*() const {
		assert(constructed == 1234);
		assert(ptr);
		return (const tree*) (ptr);
	}
#ifndef __CUDACC__
	fast_future<array<tree_ptr, NCHILD>> get_children() const;
#else
	CUDA_EXPORT
	array<tree_ptr,NCHILD> get_children() const;
#endif
	CUDA_EXPORT
	float get_radius() const;CUDA_EXPORT
	array<fixed32, NDIM> get_pos() const;CUDA_EXPORT
	bool is_leaf() const;
#ifndef __CUDACC__
	hpx::future<void> kick(kick_params_type*, bool);
#endif
};




#endif /* TREE_PTR_HPP_ */
