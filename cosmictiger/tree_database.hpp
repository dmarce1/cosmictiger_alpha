#pragma once

#include <cosmictiger/defs.hpp>
#include <cosmictiger/fixed.hpp>
#include <cosmictiger/array.hpp>
#include <cosmictiger/multipole.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/range.hpp>
class tree;
struct kick_params_type;
struct group_param_type;

size_t tree_data_bytes_used();

struct tree_ptr {
	int dindex;CUDA_EXPORT
	inline bool operator==(const tree_ptr &other) const {
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

	CUDA_EXPORT
	range get_range() const;

	CUDA_EXPORT
	void set_range(const range&) const;

#ifndef __CUDACC__
	hpx::future<size_t> find_groups(group_param_type*, bool);
	hpx::future<void> kick(kick_params_type*, bool);
#endif
};

void tree_data_initialize_kick();
void tree_data_initialize_groups();
void tree_data_free_all();
void tree_database_set_groups();

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

CUDA_EXPORT
range tree_data_get_range(int i);


CUDA_EXPORT
void tree_data_set_range(int i, const range& r);

void tree_data_clear();

std::pair<int, int> tree_data_allocate();
double tree_data_use();

class tree_allocator {
	std::pair<int, int> current_alloc;
	int next;
public:
	tree_allocator() {
		current_alloc = tree_data_allocate();
		next = current_alloc.first;
	}
	int allocate() {
		next++;
		if (next == current_alloc.second) {
			current_alloc = tree_data_allocate();
			next = current_alloc.first;
		}
		return next;
	}
};

CUDA_EXPORT
inline float tree_ptr::get_radius() const {
	return tree_data_get_radius(dindex);
}

CUDA_EXPORT
inline array<fixed32, NDIM> tree_ptr::get_pos() const {
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

CUDA_EXPORT
inline range tree_ptr::get_range() const {
	return tree_data_get_range(dindex);
}

CUDA_EXPORT
inline void tree_ptr::set_range(const range& r) const {
	tree_data_set_range(dindex, r);
}


struct multipole_pos {
	multipole multi;
	array<fixed32, NDIM> pos;
};

struct mcrit_t {
	array<fixed32, NDIM> pos;
	float radius;
};

struct tree_data_t {
	array<fixed32, NDIM> pos;
	float radius;
	array<tree_ptr, NCHILD> children;
};

struct tree_database_t {
	multipole_pos* multi;
	size_t* active_nodes;
	pair<size_t, size_t>* parts;
	tree_data_t* data;
	range* ranges;
	size_t* active_parts;
	int ntrees;
	int nchunks;
	int chunk_size;
};

#ifdef TREE_DATABASE_CU
__managed__ tree_database_t gpu_tree_data_ = {nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,1,1,1};
tree_database_t cpu_tree_data_;
#else
extern __managed__ tree_database_t gpu_tree_data_;
extern tree_database_t cpu_tree_data_;
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
	return LDG(&tree_data_.data[i].radius);
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
	tree_data_.data[i].radius = r;
}

CUDA_EXPORT inline array<fixed32, NDIM> tree_data_get_pos(int i) {
#ifdef __CUDACC__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	union fixed_union {
		fixed32 f;
		int i;
	};
	fixed_union x, y, z;
	array<fixed32, NDIM> p;
	x.i = LDG((int* )&tree_data_.data[i].pos[0]);
	y.i = LDG((int* )&tree_data_.data[i].pos[1]);
	z.i = LDG((int* )&tree_data_.data[i].pos[2]);
	p[0] = x.f;
	p[1] = y.f;
	p[2] = z.f;
	return p;
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
	tree_data_.multi[i].pos = p;
	tree_data_.data[i].pos = p;
}

CUDA_EXPORT inline multipole tree_data_get_multi(int i) {
#ifdef __CUDACC__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	multipole M;
	multipole& m = tree_data_.multi[i].multi;
	for (int i = 0; i < MP; i++) {
		M[i] = LDG(&m[i]);
	}
	return M;
}

CUDA_EXPORT inline float* tree_data_get_multi_ptr(int i) {
#ifdef __CUDACC__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	return (float*) (&tree_data_.multi[i]);
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
	return LDG(&tree_data_.data[i].children[0].dindex) == -1;

}

CUDA_EXPORT inline
void tree_data_set_isleaf(int i, bool b) {
}

CUDA_EXPORT inline array<tree_ptr, NCHILD> tree_data_get_children(int i) {
#ifdef __CUDACC__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	union child_union {
		array<tree_ptr, NCHILD> children;
		int2 ints;
	};
	child_union u;
	u.ints = LDG((int2* )(&tree_data_.data[i].children));
	return u.children;
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
	tree_data_.data[i].children[0] = c[0];
	tree_data_.data[i].children[1] = c[1];
}

CUDA_EXPORT inline pair<size_t, size_t> tree_data_get_parts(int i) {
#ifdef __CUDACC__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	union parts_union {
		pair<size_t, size_t> parts;
		int4 ints;
	};
	parts_union p;
	p.ints = LDG((int4* )(tree_data_.parts + i));
	return p.parts;
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
	return LDG(&tree_data_.active_parts[i]);
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

CUDA_EXPORT inline range tree_data_get_range(int i) {
#ifdef __CUDACC__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	return tree_data_.ranges[i];
}

CUDA_EXPORT inline
void tree_data_set_range(int i, const range& r) {
#ifdef __CUDACC__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	tree_data_.ranges[i] = r;
}

void tree_database_set_readonly();

void tree_database_unset_readonly();

