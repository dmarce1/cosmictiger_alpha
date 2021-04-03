#pragma once

#include <cosmictiger/defs.hpp>
#include <cosmictiger/fixed.hpp>
#include <cosmictiger/array.hpp>
#include <cosmictiger/multipole.hpp>
#include <cosmictiger/tree.hpp>

class tree;
struct kick_params_type;

struct tree_ptr {
	int dindex;
#ifndef NDEBUG
	int constructed;
#endif
#ifndef NDEBUG
	CUDA_EXPORT
	inline tree_ptr() {
		ptr = 0;
		constructed = 1234;
		dindex = -1;
	}
#else
	tree_ptr() = default;
#endif
	CUDA_EXPORT
	inline tree_ptr(tree_ptr &&other) {
		dindex = other.dindex;
#ifndef NDEBUG
		constructed = 1234;
#endif
	}
	CUDA_EXPORT
	inline tree_ptr(const tree_ptr &other) {
		dindex = other.dindex;
#ifndef NDEBUG
		constructed = 1234;
#endif
	}
	CUDA_EXPORT
	inline tree_ptr& operator=(const tree_ptr &other) {
		assert(constructed == 1234);
		dindex = other.dindex;
		return *this;
	}
	CUDA_EXPORT
	inline tree_ptr& operator=(tree_ptr &&other) {
		assert(constructed == 1234);
		dindex = other.dindex;
		return *this;
	}
	CUDA_EXPORT
	inline bool operator==(const tree_ptr &other) const {
		assert(constructed == 1234);
		return dindex == other.dindex;
	}

	CUDA_EXPORT
	inline array<tree_ptr, NCHILD> get_children() const;

	CUDA_EXPORT
	inline void set_children(const array<tree_ptr, NCHILD>& c) const;

	CUDA_EXPORT
	inline float get_radius() const;

	CUDA_EXPORT
	inline array<fixed32, NDIM> get_pos() const;

	CUDA_EXPORT
	inline void set_radius(float) const;

	CUDA_EXPORT
	inline void set_pos(const array<fixed32, NDIM>&) const;

	CUDA_EXPORT
	inline multipole get_multi() const;

	CUDA_EXPORT
	inline float* get_multi_ptr() const;

	CUDA_EXPORT
	inline void set_multi(const multipole&) const;

	CUDA_EXPORT
	inline bool is_leaf() const;

	CUDA_EXPORT
	inline void set_leaf(bool b) const;

	CUDA_EXPORT
	inline pair<size_t, size_t> get_parts() const;

	CUDA_EXPORT
	inline void set_parts(const pair<size_t, size_t>& p) const;

	CUDA_EXPORT
	inline size_t get_active_parts() const;

	CUDA_EXPORT
	inline void set_active_parts(size_t p) const;

	CUDA_EXPORT
	inline size_t get_active_nodes() const;

	CUDA_EXPORT
	inline void set_active_nodes(size_t p) const;

#ifndef __CUDACC__
	hpx::future<void> kick(kick_params_type*, bool);
#endif
};

void tree_data_initialize();

CUDA_EXPORT
float tree_data_get_radius(int);

CUDA_EXPORT
void tree_data_set_radius(int, float);

CUDA_EXPORT
array<fixed32, NDIM> tree_data_get_pos(int);

CUDA_EXPORT
void tree_data_set_pos(int, const array<fixed32, NDIM>&);

CUDA_EXPORT
multipole tree_data_get_multi(int i);

CUDA_EXPORT
float* tree_data_get_multi_ptr(int i);

CUDA_EXPORT
void tree_data_set_multi(int i, const multipole& m);

CUDA_EXPORT
bool tree_data_get_isleaf(int i);

CUDA_EXPORT
void tree_data_set_isleaf(int i, bool);

CUDA_EXPORT
array<tree_ptr, NCHILD> tree_data_get_children(int i);

CUDA_EXPORT
void tree_data_set_children(int i, const array<tree_ptr, NCHILD>& c);

CUDA_EXPORT
pair<size_t, size_t> tree_data_get_parts(int i);

CUDA_EXPORT
void tree_data_set_parts(int i, const pair<size_t, size_t>& p);

CUDA_EXPORT
size_t tree_data_get_active_parts(int i);

CUDA_EXPORT
void tree_data_set_active_parts(int i, size_t p);

CUDA_EXPORT
size_t tree_data_get_active_nodes(int i);

CUDA_EXPORT
void tree_data_set_active_nodes(int i, size_t p);

void tree_data_clear();

int tree_data_allocate();

CUDA_EXPORT
inline float tree_ptr::get_radius() const {
	assert(constructed == 1234);
	assert(ptr);
	return tree_data_get_radius(dindex);
}

CUDA_EXPORT
inline array<fixed32, NDIM> tree_ptr::get_pos() const {
	assert(ptr);
	assert(constructed == 1234);
	return tree_data_get_pos(dindex);
}

CUDA_EXPORT
void tree_ptr::set_radius(float r) const {
	tree_data_set_radius(dindex, r);
}

CUDA_EXPORT
void tree_ptr::set_pos(const array<fixed32, NDIM>& p) const {
	tree_data_set_pos(dindex, p);
}

CUDA_EXPORT
inline multipole tree_ptr::get_multi() const {
	return tree_data_get_multi(dindex);
}

CUDA_EXPORT
inline float* tree_ptr::get_multi_ptr() const {
	return tree_data_get_multi_ptr(dindex);
}

CUDA_EXPORT
inline void tree_ptr::set_multi(const multipole& m) const {
	tree_data_set_multi(dindex, m);
}

CUDA_EXPORT
inline bool tree_ptr::is_leaf() const {
	return tree_data_get_isleaf(dindex);
}

CUDA_EXPORT
inline void tree_ptr::set_leaf(bool b) const {
	tree_data_set_isleaf(dindex, b);
}

CUDA_EXPORT
inline array<tree_ptr, NCHILD> tree_ptr::get_children() const {
	return tree_data_get_children(dindex);
}

CUDA_EXPORT
inline void tree_ptr::set_children(const array<tree_ptr, NCHILD>& c) const {
	tree_data_set_children(dindex, c);
}

CUDA_EXPORT
inline pair<size_t, size_t> tree_ptr::get_parts() const {
	return tree_data_get_parts(dindex);
}

CUDA_EXPORT
inline void tree_ptr::set_parts(const pair<size_t, size_t>& p) const {
	tree_data_set_parts(dindex, p);
}

CUDA_EXPORT
inline size_t tree_ptr::get_active_parts() const {
	return tree_data_get_active_parts(dindex);
}

CUDA_EXPORT
inline void tree_ptr::set_active_parts(size_t p) const {
	tree_data_set_active_parts(dindex, p);
}

CUDA_EXPORT
inline size_t tree_ptr::get_active_nodes() const {
	return tree_data_get_active_nodes(dindex);
}

CUDA_EXPORT
inline void tree_ptr::set_active_nodes(size_t p) const {
	tree_data_set_active_nodes(dindex, p);
}

struct multipole_pos {
	multipole multi;
	array<fixed32, NDIM> pos;
};

struct tree_data_t {
	float* radius;
	int8_t* leaf;
	array<fixed32, NDIM>* pos;
	multipole_pos* multi;
	pair<size_t, size_t>* parts;
	array<tree_ptr, NCHILD>* children;
	size_t* active_parts;
	size_t* active_nodes;
	int ntrees;
	int nchunks;
};

#ifdef TREE_DATABASE_CU
__managed__ tree_data_t gpu_tree_data_;
tree_data_t cpu_tree_data_;
#else
extern __managed__ tree_data_t gpu_tree_data_;
extern tree_data_t cpu_tree_data_;
#endif

CUDA_EXPORT inline
float tree_data_get_radius(int i) {
#ifdef __CUDACC__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	return tree_data_.radius[i];
}

CUDA_EXPORT inline
void tree_data_set_radius(int i, float r) {
#ifdef __CUDACC__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	tree_data_.radius[i] = r;
}

CUDA_EXPORT inline array<fixed32, NDIM> tree_data_get_pos(int i) {
#ifdef __CUDACC__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	return tree_data_.pos[i];
}

CUDA_EXPORT inline
void tree_data_set_pos(int i, const array<fixed32, NDIM>& p) {
#ifdef __CUDACC__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	tree_data_.pos[i] = p;
	tree_data_.multi[i].pos = p;
}

CUDA_EXPORT inline multipole tree_data_get_multi(int i) {
#ifdef __CUDACC__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	return tree_data_.multi[i].multi;
}


CUDA_EXPORT inline float* tree_data_get_multi_ptr(int i) {
#ifdef __CUDACC__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	return (float*)(&tree_data_.multi[i]);
}

CUDA_EXPORT inline
void tree_data_set_multi(int i, const multipole& m) {
#ifdef __CUDACC__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	tree_data_.multi[i].multi = m;
}

CUDA_EXPORT inline
bool tree_data_get_isleaf(int i) {
#ifdef __CUDACC__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	return tree_data_.leaf[i];

}

CUDA_EXPORT inline
void tree_data_set_isleaf(int i, bool b) {
#ifdef __CUDACC__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	tree_data_.leaf[i] = b;

}

CUDA_EXPORT inline array<tree_ptr, NCHILD> tree_data_get_children(int i) {
#ifdef __CUDACC__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	return tree_data_.children[i];
}

CUDA_EXPORT inline
void tree_data_set_children(int i, const array<tree_ptr, NCHILD>& c) {
#ifdef __CUDACC__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	tree_data_.children[i] = c;
}

CUDA_EXPORT inline pair<size_t, size_t> tree_data_get_parts(int i) {
#ifdef __CUDACC__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	return tree_data_.parts[i];
}

CUDA_EXPORT inline
void tree_data_set_parts(int i, const pair<size_t, size_t>& p) {
#ifdef __CUDACC__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	tree_data_.parts[i] = p;
}

CUDA_EXPORT inline size_t tree_data_get_active_parts(int i) {
#ifdef __CUDACC__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	return tree_data_.active_parts[i];
}

CUDA_EXPORT inline
void tree_data_set_active_parts(int i, size_t p) {
#ifdef __CUDACC__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	tree_data_.active_parts[i] = p;
}

CUDA_EXPORT inline size_t tree_data_get_active_nodes(int i) {
#ifdef __CUDACC__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	return tree_data_.active_nodes[i];
}

CUDA_EXPORT inline
void tree_data_set_active_nodes(int i, size_t p) {
#ifdef __CUDACC__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	tree_data_.active_nodes[i] = p;
}

