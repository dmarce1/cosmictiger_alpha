#pragma once

#include <cosmictiger/defs.hpp>
#include <cosmictiger/fixed.hpp>
#include <cosmictiger/array.hpp>
#include <cosmictiger/multipole.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/range.hpp>
#include <cosmictiger/particle.hpp>

class tree;
struct kick_params_type;
struct group_param_type;

size_t tree_data_bytes_used();


using tree_use_type = int8_t;

#define TREE_KICK 0
#define TREE_GROUPS 1

struct tree_ptr {
	int dindex;
	int rank;

	bool operator<(const tree_ptr& other) const {
		if( rank < other.rank) {
			return true;
		} else if( rank > other.rank) {
			return false;
		} else {
			return dindex < other.dindex;
		}
	}

	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & dindex;
		arc & rank;
	}

	inline bool local() {
		auto prange = get_proc_range();
		return prange.second - prange.first == 1;
	}

	CUDA_EXPORT
	inline bool operator==(const tree_ptr &other) const {
		return dindex == other.dindex && rank == other.rank;
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
	inline part_iters get_parts() const;

	CUDA_EXPORT
	inline void set_parts(const part_iters& p) const;

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

	CUDA_EXPORT
	pair<int, int> get_proc_range() const;

	void set_proc_range(int, int) const;

	CUDA_EXPORT
	bool local_root() const;

	void set_local_root(bool) const;

#ifndef __CUDACC__
	hpx::future<size_t> find_groups(group_param_type*, bool);
	hpx::future<void> kick(kick_params_type*, bool);
#endif
};

struct tree_hash {
	size_t operator()(const tree_ptr& ptr) const {
		const int i = ptr.dindex  * hpx_size() + hpx_rank();
		return i;
	}
};


struct tree_node_t {
	multipole multi;
	array<fixed32, NDIM> pos;
	float radius;
	part_iters parts;
	size_t active_nodes;
	pair<int, int> proc_range;
	array<tree_ptr,NCHILD> children;
	range ranges;
	size_t active_parts;
	int8_t local_root;
	tree_use_type use;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & use;
		if (use == TREE_KICK) {
			arc & multi;
		}
		arc & children;
		arc & pos;
		arc & radius;
		arc & parts;
		arc & active_nodes;
		arc & proc_range;
		if (use == TREE_GROUPS) {
			arc & ranges;
		}
		arc & active_parts;
		arc & local_root;
	}

};

bool tree_data_read_local_root();
bool tree_data_read_cache_local_root(tree_ptr ptr);
multipole tree_data_read_cache_multi(tree_ptr ptr);
array<fixed32, NDIM> tree_data_read_cache_pos(tree_ptr ptr);
float tree_data_read_cache_radius(tree_ptr ptr);
part_iters tree_data_read_cache_parts(tree_ptr ptr);
size_t tree_data_read_cache_active_nodes(tree_ptr ptr);
pair<int, int> tree_data_read_cache_proc_range(tree_ptr ptr);
range tree_data_read_cache_range(tree_ptr ptr);
size_t tree_data_read_cache_active_parts(tree_ptr ptr);
array<tree_ptr,NCHILD> tree_data_read_cache_children(tree_ptr ptr);


void tree_data_initialize_kick();
void tree_data_initialize_groups();
void tree_data_free_all_cu();
void tree_database_set_groups();
tree_ptr tree_data_global_to_local(tree_ptr);
tree_ptr tree_data_global_to_local_recursive(tree_ptr);
void tree_data_global_to_local(stack_vector<tree_ptr>&);

void tree_data_initialize(tree_use_type);
void tree_data_free_all();

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
part_iters tree_data_get_parts(int i, int pi);

CUDA_EXPORT
part_iters tree_data_get_parts(int i);

CUDA_EXPORT
void tree_data_set_parts(int i, const part_iters& p);

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

CUDA_EXPORT pair<int, int> tree_data_get_proc_range(int i);

void tree_data_set_proc_range(int, int, int);

CUDA_EXPORT bool tree_data_local_root(int i);

void tree_data_set_local_root(int, bool);

void tree_data_clear();
void tree_data_set_cache_line_size();
void tree_data_clear_cu();

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

CUDA_EXPORT int hpx_rank_cuda();

CUDA_EXPORT
inline float tree_ptr::get_radius() const {
#ifndef __CUDACC__
	if (rank == hpx_rank()) {
#endif
		assert(rank == hpx_rank_cuda());
		return tree_data_get_radius(dindex);
#ifndef __CUDACC__
	} else {
		return tree_data_read_cache_radius(*this);
	}
#endif
}

CUDA_EXPORT
inline array<fixed32, NDIM> tree_ptr::get_pos() const {
#ifndef __CUDACC__
	if (rank == hpx_rank()) {
#endif
		assert(rank == hpx_rank_cuda());
		return tree_data_get_pos(dindex);
#ifndef __CUDACC__
	} else {
		return tree_data_read_cache_pos(*this);
	}
#endif
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
#ifndef __CUDACC__
	if (rank == hpx_rank()) {
#endif
		assert(rank == hpx_rank_cuda());
		return tree_data_get_multi(dindex);
#ifndef __CUDACC__
	} else {
		return tree_data_read_cache_multi(*this);
	}
#endif
}

CUDA_EXPORT
inline float* tree_ptr::get_multi_ptr() const {
	assert(rank == hpx_rank_cuda());
	return tree_data_get_multi_ptr(dindex);
}

CUDA_EXPORT
inline void tree_ptr::set_multi(const multipole& m) const {
	assert(rank == hpx_rank_cuda());
	tree_data_set_multi(dindex, m);
}

CUDA_EXPORT
inline bool tree_ptr::is_leaf() const {
#ifndef __CUDACC__
	if (rank == hpx_rank()) {
#endif
		assert(rank == hpx_rank_cuda());
		return tree_data_get_isleaf(dindex);
#ifndef __CUDACC__
	} else {
		return tree_data_read_cache_children(*this)[0].dindex == -1;
	}
#endif
}

CUDA_EXPORT
inline void tree_ptr::set_leaf(bool b) const {
	assert(rank == hpx_rank_cuda());
	tree_data_set_isleaf(dindex, b);
}

CUDA_EXPORT
inline array<tree_ptr, NCHILD> tree_ptr::get_children() const {
#ifndef __CUDACC__
	if (rank == hpx_rank()) {
#endif
		assert(rank == hpx_rank_cuda());
		return tree_data_get_children(dindex);
#ifndef __CUDACC__
	} else {
		return tree_data_read_cache_children(*this);
	}
#endif
}

CUDA_EXPORT
inline void tree_ptr::set_children(const array<tree_ptr, NCHILD>& c) const {
	tree_data_set_children(dindex, c);
}

CUDA_EXPORT
inline part_iters tree_ptr::get_parts() const {
#ifndef __CUDACC__
	if (rank == hpx_rank()) {
#endif
		assert(rank == hpx_rank_cuda());
		return tree_data_get_parts(dindex);
#ifndef __CUDACC__
	} else {
		return tree_data_read_cache_parts(*this);
	}
#endif
}

CUDA_EXPORT
inline void tree_ptr::set_parts(const part_iters& p) const {
	tree_data_set_parts(dindex, p);
}

CUDA_EXPORT
inline size_t tree_ptr::get_active_parts() const {
#ifndef __CUDACC__
	if (rank == hpx_rank()) {
#endif
		assert(rank == hpx_rank_cuda());
		return tree_data_get_active_parts(dindex);
#ifndef __CUDACC__
	} else {
		return tree_data_read_cache_active_parts(*this);
	}
#endif
}

CUDA_EXPORT
inline void tree_ptr::set_active_parts(size_t p) const {
	tree_data_set_active_parts(dindex, p);
}

CUDA_EXPORT
inline size_t tree_ptr::get_active_nodes() const {
#ifndef __CUDACC__
	if (rank == hpx_rank()) {
#endif
		assert(rank == hpx_rank_cuda());
		return tree_data_get_active_nodes(dindex);
#ifndef __CUDACC__
	} else {
		return tree_data_read_cache_active_nodes(*this);
	}
#endif
}

CUDA_EXPORT
inline void tree_ptr::set_active_nodes(size_t p) const {
	tree_data_set_active_nodes(dindex, p);
}

CUDA_EXPORT
inline range tree_ptr::get_range() const {
#ifndef __CUDACC__
	if (rank == hpx_rank()) {
#endif
		assert(rank == hpx_rank_cuda());
		return tree_data_get_range(dindex);
#ifndef __CUDACC__
	} else {
		return tree_data_read_cache_range(*this);
	}
#endif
}

CUDA_EXPORT
inline void tree_ptr::set_range(const range& r) const {
	tree_data_set_range(dindex, r);
}

CUDA_EXPORT
inline pair<int, int> tree_ptr::get_proc_range() const {
#ifndef __CUDACC__
	if (rank == hpx_rank()) {
#endif
		assert(rank == hpx_rank_cuda());
		return tree_data_get_proc_range(dindex);
#ifndef __CUDACC__
	} else {
		return tree_data_read_cache_proc_range(*this);
	}
#endif
}

inline void tree_ptr::set_local_root(bool b) const {
	tree_data_set_local_root(dindex, b);
}

void tree_data_free_cache();


CUDA_EXPORT
inline bool tree_ptr::local_root() const {
#ifndef __CUDACC__
	if (rank == hpx_rank()) {
#endif
		assert(rank == hpx_rank_cuda());
		return tree_data_local_root(dindex);
#ifndef __CUDACC__
	} else {
		return tree_data_read_cache_local_root(*this);
	}
#endif
}

inline void tree_ptr::set_proc_range(int b, int e) const {
	tree_data_set_proc_range(dindex, b, e);
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
	part_iters* parts;
	tree_data_t* data;
	pair<int, int>* proc_range;
	range* ranges;
	size_t* active_parts;
	int8_t* local_root;
	int ntrees;
	int nchunks;
	int chunk_size;
};

#ifdef TREE_DATABASE_CU
__constant__ tree_database_t gpu_tree_data_ = {nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,1,1,1};
tree_database_t cpu_tree_data_;
#else
extern __constant__ tree_database_t gpu_tree_data_;
extern tree_database_t cpu_tree_data_;
#endif

CUDA_EXPORT inline
float tree_data_get_radius(int i) {
#ifdef __CUDA_ARCH__
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
#ifdef __CUDA_ARCH__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	tree_data_.data[i].radius = r;
}

CUDA_EXPORT inline array<fixed32, NDIM> tree_data_get_pos(int i) {
#ifdef __CUDA_ARCH__
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
#ifdef __CUDA_ARCH__
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
#ifdef __CUDA_ARCH__
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
#ifdef __CUDA_ARCH__
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
#ifdef __CUDA_ARCH__
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
#ifdef __CUDA_ARCH__
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
#ifdef __CUDA_ARCH__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	union child_union {
		array<tree_ptr, NCHILD> children;
		int4 ints;
	};
	child_union u;
	u.ints = LDG((int4* )(&tree_data_.data[i].children));
	return u.children;
}

CUDA_EXPORT inline
void tree_data_set_children(int i, const array<tree_ptr, NCHILD>& c) {
#ifdef __CUDA_ARCH__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	tree_data_.data[i].children[0] = c[0];
	tree_data_.data[i].children[1] = c[1];
}

CUDA_EXPORT inline part_iters tree_data_get_parts(int i) {
#ifdef __CUDA_ARCH__
	auto& tree_data_ = gpu_tree_data_;
#ifndef NDEBUG
	if( tree_data_get_isleaf(i)) {
		assert(tree_data_.proc_range[i].first == hpx_rank_cuda());
	}
#endif
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	union parts_union {
		part_iters parts;
		int2 ints;
	};
	parts_union p;
	p.ints = LDG((int2* )(&tree_data_.parts[i]));

	return p.parts;
}

CUDA_EXPORT inline
void tree_data_set_parts(int i, const part_iters& p) {
#ifdef __CUDA_ARCH__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	tree_data_.parts[i] = p;
}

CUDA_EXPORT inline size_t tree_data_get_active_parts(int i) {
#ifdef __CUDA_ARCH__
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
#ifdef __CUDA_ARCH__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	tree_data_.active_parts[i] = p;
}

CUDA_EXPORT inline size_t tree_data_get_active_nodes(int i) {
#ifdef __CUDA_ARCH__
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
#ifdef __CUDA_ARCH__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	tree_data_.active_nodes[i] = p;
}

CUDA_EXPORT inline range tree_data_get_range(int i) {
#ifdef __CUDA_ARCH__
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
#ifdef __CUDA_ARCH__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	tree_data_.ranges[i] = r;
}

CUDA_EXPORT inline pair<int, int> tree_data_get_proc_range(int i) {
#ifdef __CUDA_ARCH__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	return tree_data_.proc_range[i];
}

inline
void tree_data_set_proc_range(int i, int b, int e) {
#ifdef __CUDA_ARCH__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	tree_data_.proc_range[i].first = b;
	tree_data_.proc_range[i].second = e;
}
CUDA_EXPORT inline bool tree_data_local_root(int i) {
#ifdef __CUDA_ARCH__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	return tree_data_.local_root[i];
}

inline
void tree_data_set_local_root(int i, bool b) {
#ifdef __CUDA_ARCH__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	tree_data_.local_root[i]= b;
}

void tree_data_map_global_to_local1();

void tree_data_map_global_to_local2();

