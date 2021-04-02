/*
 * tree_ptr_impl.hpp
 *
 *  Created on: Apr 1, 2021
 *      Author: dmarce1
 */

#ifndef TREE_PTR_IMPL_HPP_
#define TREE_PTR_IMPL_HPP_

#ifndef TREE_CU
extern __managed__ trees cuda_trees_database;
#endif

CUDA_EXPORT
inline multi_crit tree_ptr::get_mcrit() const {
	assert(index !=-1234);
/*	multi_crit m;
	m.r = ((tree*)(ptr))->radius;
	m.pos= ((tree*)(ptr))->pos;
	return m;
*/
#ifdef __CUDA_ARCH__
	const trees& tree_data = cuda_trees_database;
#else
	const trees& tree_data = trees_allocator::get_trees();
#endif
	const auto mcrit = tree_data.get_mcrit(index);
	return mcrit;
}

CUDA_EXPORT
inline void tree_ptr::set_mcrit(const array<fixed32, NDIM>& p, float r) {
	assert(index !=-1234);
/*	((tree*)(ptr))->radius = r;
	((tree*)(ptr))->pos = p;
	return;*/
#ifdef __CUDA_ARCH__
	trees& tree_data = cuda_trees_database;
#else
	trees& tree_data = trees_allocator::get_trees();
#endif
	tree_data.set_mcrit(index, p, r);
}

CUDA_EXPORT
inline const multipole_pos& tree_ptr::get_mpole() const {
	assert(index !=-1234);
#ifdef __CUDA_ARCH__
	const trees& tree_data = cuda_trees_database;
#else
	const trees& tree_data = trees_allocator::get_trees();
#endif
	return tree_data.get_multi(index);

}

CUDA_EXPORT
inline void tree_ptr::set_mpole(const multipole& m, const array<fixed32, NDIM>& p) {
	assert(index !=-1234);
#ifdef __CUDA_ARCH__
	trees& tree_data = cuda_trees_database;
#else
	trees& tree_data = trees_allocator::get_trees();
#endif
	return tree_data.set_multi(index, m, p);
}

CUDA_EXPORT
inline parts_type tree_ptr::get_parts() const {
	assert(index !=-1234);
	return ((tree*) (ptr))->parts;
}

CUDA_EXPORT
inline children_type tree_ptr::get_children() const {
	assert(index !=-1234);
	return ((tree*) (ptr))->children;
}

CUDA_EXPORT
inline active_type tree_ptr::get_active() const {
	assert(index !=-1234);
	active_type a;
	a.nodes = ((tree*) (ptr))->active_nodes;
	a.parts = ((tree*) (ptr))->active_parts;
	return a;
}

#endif /* TREE_PTR_IMPL_HPP_ */
