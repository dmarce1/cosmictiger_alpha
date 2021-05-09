#pragma once

#include <cosmictiger/defs.hpp>
#include <cosmictiger/fixed.hpp>
#include <cosmictiger/array.hpp>
#include <cosmictiger/multipole.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/range.hpp>

#include <cosmictiger/hpx.hpp>

#include <memory>

class tree;
struct kick_params_type;
struct group_param_type;
struct sph_neighbor_params_type;

size_t tree_data_bytes_used();

struct tree_ptr {
	int rank;
	int dindex;

	template<class A>
	void serialize(A && arc, unsigned) {
		arc & rank;
		arc & dindex;
	}

	CUDA_EXPORT
	inline bool operator==(const tree_ptr &other) const {
		return dindex == other.dindex && rank == other.rank;
	}

	CUDA_EXPORT
	inline array<tree_ptr, NCHILD> get_children() const;

	inline void set_children(const array<tree_ptr, NCHILD>& c) const;

	inline bool all_local() const;

	void set_all_local(bool) const;

	CUDA_EXPORT
	inline float get_radius() const;

	CUDA_EXPORT
	inline array<fixed32, NDIM> get_pos() const;

	inline void set_radius(float) const;

	inline void set_pos(const array<fixed32, NDIM>&) const;

	CUDA_EXPORT
	inline multipole get_multi() const;

	CUDA_EXPORT
	inline float* get_multi_ptr() const;

	inline void set_multi(const multipole&) const;

	CUDA_EXPORT
	inline bool is_leaf() const;

	inline void set_leaf(bool b) const;

	CUDA_EXPORT
	inline part_iters get_parts(int) const;

	CUDA_EXPORT
	inline parts_type get_parts() const;

	inline void set_parts(const parts_type& p) const;

	CUDA_EXPORT
	inline size_t get_active_parts() const;

	inline void set_active_parts(size_t p) const;

	CUDA_EXPORT
	inline size_t get_active_nodes() const;

	CUDA_EXPORT
	inline void set_active_nodes(size_t p) const;

	CUDA_EXPORT
	range get_range() const;

	CUDA_EXPORT
	range get_sph_range() const;

	void set_range(const range&) const;

	void set_sph_range(const range&) const;

#ifndef __CUDACC__
	hpx::future<size_t> find_groups(group_param_type*, bool);
	hpx::future<void> kick(kick_params_type*, bool);
	hpx::future<bool> sph_neighbors(sph_neighbor_params_type*, bool);
#endif
};

enum tree_use_type {
	KICK, GROUP
};

void tree_data_initialize(tree_use_type use_type);
void tree_data_free_all();
void tree_data_set_groups();

bool tree_data_all_local(int);

void tree_data_set_all_local(int, bool);

CUDA_EXPORT
float tree_data_get_radius(int);

CUDA_EXPORT
array<fixed32, NDIM> tree_data_get_pos(int);

CUDA_EXPORT
multipole tree_data_get_multi(int i);

CUDA_EXPORT
bool tree_data_get_isleaf(int i);

CUDA_EXPORT
array<tree_ptr, NCHILD> tree_data_get_children(int i);

CUDA_EXPORT
part_iters tree_data_get_parts(int i, int pi);

CUDA_EXPORT
size_t tree_data_get_active_parts(int i);

CUDA_EXPORT
size_t tree_data_get_active_nodes(int i);

CUDA_EXPORT
range tree_data_get_range(int i);

CUDA_EXPORT
range tree_data_get_sph_range(int i);

CUDA_EXPORT
float* tree_data_get_multi_ptr(int i);

void tree_data_set_radius(int, float);

void tree_data_set_pos(int, const array<fixed32, NDIM>&);

void tree_data_set_multi(int i, const multipole& m);

void tree_data_set_isleaf(int i, bool);

void tree_data_set_children(int i, const array<tree_ptr, NCHILD>& c);

CUDA_EXPORT
parts_type tree_data_get_parts(int i);

void tree_data_set_parts(int i, const parts_type& p);

void tree_data_set_active_parts(int i, size_t p);

CUDA_EXPORT
void tree_data_set_active_nodes(int i, size_t p);

void tree_data_set_range(int i, const range& r);

void tree_data_set_sph_range(int i, const range& r);

void tree_data_clear();

std::pair<int, int> tree_data_allocate();
double tree_data_use();

class tree_allocator {
	std::pair<int, int> current_alloc;
	int next;
public:
	tree_allocator();
	tree_ptr allocate();
};

#ifndef __CUDA_ARCH__
float tree_cache_get_radius(tree_ptr ptr);
bool tree_cache_get_all_local(tree_ptr ptr);
array<fixed32, NDIM> tree_cache_get_pos(tree_ptr ptr);
multipole tree_cache_get_multi(tree_ptr ptr);
bool tree_cache_get_isleaf(tree_ptr ptr);
array<tree_ptr, NCHILD> tree_cache_get_children(tree_ptr ptr);
part_iters tree_cache_get_parts(tree_ptr ptr, int pi);
parts_type tree_cache_get_parts(tree_ptr ptr);
size_t tree_cache_get_active_parts(tree_ptr ptr);
size_t tree_cache_get_active_nodes(tree_ptr ptr);
range tree_cache_get_range(tree_ptr ptr);
range tree_cache_get_sph_range(tree_ptr ptr);
#endif

inline bool tree_ptr::all_local() const {
#ifndef __CUDACC__
	if (rank == hpx_rank()) {
#endif
		return tree_data_all_local(dindex);
#ifndef __CUDACC__
	} else {
		return tree_cache_get_all_local(*this);
	}
#endif
}

inline void tree_ptr::set_all_local(bool b) const {
	tree_data_set_all_local(dindex, b);
}

CUDA_EXPORT
inline float tree_ptr::get_radius() const {
#ifndef __CUDACC__
	if (rank == hpx_rank()) {
#endif
		return tree_data_get_radius(dindex);
#ifndef __CUDACC__
	} else {
		return tree_cache_get_radius(*this);
	}
#endif
}

CUDA_EXPORT
inline array<fixed32, NDIM> tree_ptr::get_pos() const {
#ifndef __CUDACC__
	if (rank == hpx_rank()) {
#endif
		return tree_data_get_pos(dindex);
#ifndef __CUDACC__
	} else {
		return tree_cache_get_pos(*this);
	}
#endif
}

void tree_ptr::set_radius(float r) const {
	tree_data_set_radius(dindex, r);
}

void tree_ptr::set_pos(const array<fixed32, NDIM>& p) const {
	tree_data_set_pos(dindex, p);
}

CUDA_EXPORT
inline multipole tree_ptr::get_multi() const {
#ifndef __CUDACC__
	if (rank == hpx_rank()) {
#endif
		return tree_data_get_multi(dindex);
#ifndef __CUDACC__
	} else {
		return tree_cache_get_multi(*this);
	}
#endif
}

CUDA_EXPORT
inline float* tree_ptr::get_multi_ptr() const {
	return tree_data_get_multi_ptr(dindex);
}

inline void tree_ptr::set_multi(const multipole& m) const {
	tree_data_set_multi(dindex, m);
}

CUDA_EXPORT
inline bool tree_ptr::is_leaf() const {
#ifndef __CUDACC__
	if (rank == hpx_rank()) {
#endif
		return tree_data_get_isleaf(dindex);
#ifndef __CUDACC__
	} else {
		return tree_cache_get_isleaf(*this);
	}
#endif
}

inline void tree_ptr::set_leaf(bool b) const {
	tree_data_set_isleaf(dindex, b);
}

CUDA_EXPORT
inline array<tree_ptr, NCHILD> tree_ptr::get_children() const {
#ifndef __CUDACC__
	if (rank == hpx_rank()) {
#endif
		return tree_data_get_children(dindex);
#ifndef __CUDACC__
	} else {
		return tree_cache_get_children(*this);
	}
#endif
}

inline void tree_ptr::set_children(const array<tree_ptr, NCHILD>& c) const {
	tree_data_set_children(dindex, c);
}

CUDA_EXPORT
inline part_iters tree_ptr::get_parts(int pi) const {
#ifndef __CUDACC__
	if (rank == hpx_rank()) {
#endif
		return tree_data_get_parts(dindex, pi);
#ifndef __CUDACC__
	} else {
		return tree_cache_get_parts(*this, pi);
	}
#endif
}

CUDA_EXPORT
inline parts_type tree_ptr::get_parts() const {
#ifndef __CUDACC__
	if (rank == hpx_rank()) {
#endif
		return tree_data_get_parts(dindex);
#ifndef __CUDACC__
	} else {
		return tree_cache_get_parts(*this);
	}
#endif
}

inline void tree_ptr::set_parts(const parts_type& p) const {
	tree_data_set_parts(dindex, p);
}

CUDA_EXPORT
inline size_t tree_ptr::get_active_parts() const {
#ifndef __CUDACC__
	if (rank == hpx_rank()) {
#endif
		return tree_data_get_active_parts(dindex);
#ifndef __CUDACC__
	} else {
		return tree_cache_get_active_parts(*this);
	}
#endif
}

inline void tree_ptr::set_active_parts(size_t p) const {
	tree_data_set_active_parts(dindex, p);
}

CUDA_EXPORT
inline size_t tree_ptr::get_active_nodes() const {
#ifndef __CUDACC__
	if (rank == hpx_rank()) {
#endif
		return tree_data_get_active_nodes(dindex);
#ifndef __CUDACC__
	} else {
		return tree_cache_get_active_nodes(*this);
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
		return tree_data_get_range(dindex);
#ifndef __CUDACC__
	} else {
		return tree_cache_get_range(*this);
	}
#endif
}

inline void tree_ptr::set_range(const range& r) const {
	tree_data_set_range(dindex, r);
}

CUDA_EXPORT
inline range tree_ptr::get_sph_range() const {
#ifndef __CUDACC__
	if (rank == hpx_rank()) {
#endif
		return tree_data_get_sph_range(dindex);
#ifndef __CUDACC__
	} else {
		return tree_cache_get_sph_range(*this);
	}
#endif
}

inline void tree_ptr::set_sph_range(const range& r) const {
	tree_data_set_sph_range(dindex, r);
}

struct multipole_pos {
	multipole multi;
	array<fixed32, NDIM> pos;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & multi;
		arc & pos;
	}
};

struct mcrit_t {
	array<fixed32, NDIM> pos;
	float radius;
};

struct tree_data_t {
	array<fixed32, NDIM> pos;
	float radius;
	array<tree_ptr, NCHILD> children;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & pos;
		arc & radius;
		arc & children;
	}
};

struct tree_database_t {
	multipole_pos* multi;
	size_t* active_nodes;
	parts_type* parts;
	tree_data_t* data;
	range* ranges;
	range* sph_ranges;
	size_t* active_parts;
	bool* all_local;
	int ntrees;
	int nchunks;
	int chunk_size;

#ifndef __CUDACC__
	HPX_SERIALIZATION_SPLIT_MEMBER();

	template<class A>
	void save(A&& arc, unsigned) const {
		arc & bool(multi);
		arc & bool(ranges);
		arc & bool(sph_ranges);
		arc & ntrees;
		arc & nchunks;
		arc & chunk_size;
		for (int i = 0; i < ntrees; i++) {
			if (multi) {
				arc & multi[i];
			}
			arc & active_nodes[i];
			arc & parts[i];
			arc & data[i];
			if (ranges) {
				arc & ranges[i];
			}
			if (sph_ranges) {
				arc & ranges[i];
			}
			arc & active_parts[i];
			arc & all_local[i];
		}
		delete[] data;
		delete[] parts;
		delete[] active_nodes;
		delete[] active_parts;
		delete[] all_local;
		if (multi) {
			delete[] multi;
		}
		if (ranges) {
			delete[] ranges;
		}
		if (sph_ranges) {
			delete[] sph_ranges;
		}

	}

	template<class A>
	void load(A&& arc, unsigned) {
		bool has_multi, has_ranges, has_sph_ranges;
		arc & has_multi;
		arc & has_ranges;
		arc & has_sph_ranges;
		arc & ntrees;
		arc & nchunks;
		arc & chunk_size;
		data = new tree_data_t[ntrees];
		parts = new parts_type[ntrees];
		active_nodes = new size_t[ntrees];
		active_parts = new size_t[ntrees];
		all_local = new bool[ntrees];
		if (has_multi) {
			multi = new multipole_pos[ntrees];
		} else {
			multi = nullptr;
		}
		if (has_ranges) {
			ranges = new range[ntrees];
		} else {
			ranges = nullptr;
		}
		if (has_sph_ranges) {
			sph_ranges = new range[ntrees];
		} else {
			sph_ranges = nullptr;
		}
		for (int i = 0; i < ntrees; i++) {
			if (multi) {
				arc & multi[i];
			}
			arc & active_nodes[i];
			arc & parts[i];
			arc & data[i];
			if (ranges) {
				arc & ranges[i];
			}
			if (sph_ranges) {
				arc & ranges[i];
			}
			arc & active_parts[i];
			arc & all_local[i];
		}
	}
#endif
};



#ifdef TREE_DATABASE_CU
__constant__ tree_database_t gpu_tree_data_;
tree_database_t cpu_tree_data_ = {nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,1,1,1};
#else
extern __constant__ tree_database_t gpu_tree_data_;
extern tree_database_t cpu_tree_data_;
#endif

inline bool tree_data_all_local(int i) {
	auto& tree_data_ = cpu_tree_data_;
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	return tree_data_.all_local[i];
}

inline void tree_data_set_all_local(int i, bool b) {
	auto& tree_data_ = cpu_tree_data_;
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	tree_data_.all_local[i] = b;
}

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

inline
void tree_data_set_radius(int i, float r) {
	auto& tree_data_ = cpu_tree_data_;
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

inline
void tree_data_set_pos(int i, const array<fixed32, NDIM>& p) {
	auto& tree_data_ = cpu_tree_data_;
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

inline
void tree_data_set_multi(int i, const multipole& m) {
	auto& tree_data_ = cpu_tree_data_;
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

inline
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
		int4 ints;
	};
	child_union u;
	u.ints = LDG((int4* )(&tree_data_.data[i].children));
	return u.children;
}

inline
void tree_data_set_children(int i, const array<tree_ptr, NCHILD>& c) {
	auto& tree_data_ = cpu_tree_data_;
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	tree_data_.data[i].children[0] = c[0];
	tree_data_.data[i].children[1] = c[1];
}

CUDA_EXPORT inline part_iters tree_data_get_parts(int i, int pi) {
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
	p.ints = LDG((int4* )(&tree_data_.parts[i][pi]));
	return p.parts;
}

CUDA_EXPORT inline parts_type tree_data_get_parts(int i) {
	parts_type parts;
	for (int pi = 0; pi < NPART_TYPES; pi++) {
		parts[pi] = tree_data_get_parts(i, pi);
	}
	return parts;
}

inline
void tree_data_set_parts(int i, const parts_type& p) {
	auto& tree_data_ = cpu_tree_data_;
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

inline
void tree_data_set_active_parts(int i, size_t p) {
	auto& tree_data_ = cpu_tree_data_;
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

inline
void tree_data_set_range(int i, const range& r) {
	auto& tree_data_ = cpu_tree_data_;
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	tree_data_.ranges[i] = r;
}

CUDA_EXPORT inline range tree_data_get_sph_range(int i) {
#ifdef __CUDACC__
	auto& tree_data_ = gpu_tree_data_;
#else
	auto& tree_data_ = cpu_tree_data_;
#endif
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	return tree_data_.sph_ranges[i];
}

inline
void tree_data_set_sph_range(int i, const range& r) {
	auto& tree_data_ = cpu_tree_data_;
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	tree_data_.sph_ranges[i] = r;
}

void tree_database_set_readonly();
void tree_database_unset_readonly();
tree_database_t tree_cache_line_fetch(int index);
void tree_cache_clear();

