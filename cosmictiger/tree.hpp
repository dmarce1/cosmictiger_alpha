#pragma once

#include <cosmictiger/defs.hpp>
#include <cosmictiger/particle.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/multipole.hpp>
#include <cosmictiger/fast_future.hpp>
#include <cosmictiger/expansion.hpp>
#include <cosmictiger/lockfree_queue.hpp>
#include <cosmictiger/interactions.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/stack_vector.hpp>

#include <functional>

#include <queue>
#include <memory>
#include <stack>

#define LEFT 0
#define RIGHT 1
//#define KICK_CUDA_SIZE (1<<16)

#define TREE_PTR_STACK (TREE_MAX_DEPTH*WORKSPACE_SIZE)

#define EWALD_MIN_DIST2 (0.25f * 0.25f)

class tree;
struct sort_params;
struct tree_ptr;

#ifndef __CUDACC__
struct tree_alloc {
	managed_allocator<multipole> multi_alloc;
	managed_allocator<tree> tree_alloc;
};

struct sort_params {
#ifdef TEST_STACK
	uint8_t* stack_ptr;
#endif
	range box;
	std::shared_ptr<std::vector<size_t>> bounds;
	std::shared_ptr<tree_alloc> allocs;
	uint32_t key_begin;
	uint32_t key_end;
	int8_t depth;
	int8_t min_depth;

	template<class A>
	void serialization(A &&arc, unsigned) {
		/********* ADD******/
	}

	sort_params() {
		depth = -1;
	}

	bool iamroot() const {
		return depth == -1;
	}

	void set_root() {
		const auto opts = global().opts;
		for (int dim = 0; dim < NDIM; dim++) {
			box.begin[dim] = 0.f;
			box.end[dim] = 1.f;
		}
#ifdef TEST_STACK
		stack_ptr = (uint8_t*) &stack_ptr;
#endif
		depth = 0;
		bounds = std::make_shared<std::vector<size_t>>(2);
		(*bounds)[0] = 0;
		(*bounds)[1] = opts.nparts;
		key_begin = 0;
		key_end = 1;
		allocs = std::make_shared<tree_alloc>();
	}

	std::pair<size_t, size_t> get_bounds() const {
		std::pair<size_t, size_t> rc;
		rc.first = (*bounds)[key_begin];
		rc.second = (*bounds)[key_end];
		return rc;
	}

	std::array<sort_params, NCHILD> get_children() const {
		std::array<sort_params, NCHILD> child;
		for (int i = 0; i < NCHILD; i++) {
			child[i].bounds = bounds;
			child[i].depth = depth + 1;
			child[i].allocs = allocs;
			child[i].box = box;
#ifdef TEST_STACK
			child[i].stack_ptr = stack_ptr;
#endif
		}
		int sort_dim = depth % NDIM;
		child[LEFT].box.end[sort_dim] = child[RIGHT].box.begin[sort_dim] = (fixed64(box.begin[sort_dim])
				+ fixed64(box.end[sort_dim])) / fixed64(2);
		child[LEFT].key_begin = key_begin;
		child[LEFT].key_end = child[RIGHT].key_begin = ((key_begin + key_end) >> 1);
		child[RIGHT].key_end = key_end;
		return child;
	}
};

#endif

struct kick_return {
	int8_t rung;
	size_t flops;
};

class tree_ptr;
class kick_params_type;

struct tree_ptr {
	uintptr_t ptr;
//   int rank;
//	int8_t opened;
#ifndef NDEBUG
	int constructed;
#endif
	CUDA_EXPORT
	inline tree_ptr() {
		//   rank = -1;
		ptr = 0;
//		opened = false;
#ifndef NDEBUG
		constructed = 1234;
#endif
	}
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
	hpx::future<kick_return> kick(kick_params_type*, bool, bool);
#endif
};

struct sort_return {
	tree_ptr check;
	template<class A>
	void serialization(A &&arc, unsigned) {
		assert(false);
	}
};
template<class A, class B>
struct pair {
	A first;
	B second;
};

#define NITERS 4
struct cuda_ewald_shmem {
	array<float, KICK_BLOCK_SIZE> Lreduce;  // 256
};

struct cuda_kick_shmem {
	union {
		array<array<float, KICK_BLOCK_SIZE>, NDIM> f; // 384
		struct {
			array<int16_t, NITERS> count; // 8
			array<array<int16_t, KICK_BLOCK_SIZE + 1>, NITERS> indices; //33
		};
		array<float, KICK_BLOCK_SIZE> Lreduce;  // 4480
	};
	array<array<fixed32, KICK_PP_MAX>, NDIM> src;  // 3072
	array<array<fixed32, MAX_BUCKET_SIZE>, NDIM> sink;  // 768
	array<int8_t, MAX_BUCKET_SIZE> rungs; // 256
};

struct kick_params_type {
	vector<tree_ptr> multi_interactions;
	vector<tree_ptr> part_interactions;
	vector<tree_ptr> tmp;
	vector<tree_ptr> next_checks;
	vector<tree_ptr> opened_checks;
	stack_vector<tree_ptr> dchecks;
	stack_vector<tree_ptr> echecks;
	array<array<float, MAX_BUCKET_SIZE>, NDIM> F;
	array<expansion<float>, TREE_MAX_DEPTH> L;
	array<array<fixed32, NDIM>, TREE_MAX_DEPTH> Lpos;
	tree_ptr tptr;
	int depth;
	int cuda_cutoff;
	float theta;
	float eta;
	float scale;
	float hsoft;
	int rung;
	bool t0;
	size_t flops;CUDA_EXPORT
	inline kick_params_type() {
		THREAD;
		if (tid == 0) {
			depth = 0;
			theta = 0.4;
			eta = 0.1;
			scale = 1.0;
			t0 = 1.0;
			rung = 0;
			hsoft = global().opts.hsoft;
			theta = global().opts.theta;
		}CUDA_SYNC();
	}
	friend class tree_ptr;
};

struct kick_params_type;

#ifndef __CUDACC__
struct gpu_kick {
	kick_params_type *params;
	hpx::lcos::local::promise<kick_return> promise;
	pair<size_t, size_t> parts;
};

struct gpu_ewald {
	kick_params_type *params;
	hpx::lcos::local::promise<int32_t> promise;
};
#endif

struct tree {

#ifndef __CUDACC__
private:
#endif
	array<fixed32, NDIM> pos;
	float radius;
	array<tree_ptr, NCHILD> children;
	pair<size_t, size_t> parts;
	multipole multi;
	static ewald_indices* real_indices_ptr;
	static ewald_indices* four_indices_ptr;
	static periodic_parts* periodic_parts_ptr;

public:
	static particle_set *particles;
	static std::atomic<int> cuda_node_count;
	static std::atomic<int> cpu_node_count;
	static void set_cuda_particle_set(particle_set*);
	static void cuda_set_kick_params(particle_set *p, ewald_indices *four_indices, ewald_indices *real_indices,
			periodic_parts *periodic_parts);
#ifndef __CUDACC__
	static void set_particle_set(particle_set*);
	inline static hpx::future<sort_return> create_child(sort_params&);
	static fast_future<sort_return> cleanup_child();
	static hpx::lcos::local::mutex mtx;
	static hpx::lcos::local::mutex gpu_mtx;
	hpx::future<kick_return> send_kick_to_gpu(kick_params_type *params);
	hpx::future<int32_t> send_ewald_to_gpu(kick_params_type *params);
	static void gpu_daemon();
	inline bool is_leaf() const {
		return children[0] == tree_ptr();
	}
	static void cleanup();
	int cpu_cc_direct(kick_params_type *params);
	int cpu_pp_direct(kick_params_type *params);
	int cpu_cc_ewald(kick_params_type *params);
	sort_return sort(sort_params = sort_params());
	hpx::future<kick_return> kick(kick_params_type*);
	static std::atomic<bool> daemon_running;
	static std::atomic<bool> shutdown_daemon;
	static lockfree_queue<gpu_kick, GPU_QUEUE_SIZE> gpu_queue;
	static lockfree_queue<gpu_ewald, GPU_QUEUE_SIZE> gpu_ewald_queue;
#endif
	int compute_block_count(size_t cutoff);
	friend class tree_ptr;
};

#ifndef __CUDACC__
inline fast_future<array<tree_ptr, NCHILD>> tree_ptr::get_children() const {
	assert(constructed == 1234);
	assert(ptr);
	return fast_future<array<tree_ptr, NCHILD>>(((tree*) ptr)->children);
}
#else
CUDA_EXPORT
inline array<tree_ptr, NCHILD> tree_ptr::get_children() const {
	assert(constructed == 1234);
	assert(ptr);
	return (((tree*) ptr)->children);
}

#endif

CUDA_EXPORT
inline float tree_ptr::get_radius() const {
	assert(constructed == 1234);
	assert(ptr);
	return ((tree*) ptr)->radius;
}

CUDA_EXPORT
inline array<fixed32, NDIM> tree_ptr::get_pos() const {
	assert(ptr);
	assert(constructed == 1234);
	return ((tree*) ptr)->pos;
}

CUDA_EXPORT
inline bool tree_ptr::is_leaf() const {
	assert(constructed == 1234);
	assert(ptr);
	return ((tree*) ptr)->children[0] == tree_ptr();
}

cudaStream_t get_stream();
void cleanup_stream(cudaStream_t s);

std::function<bool()> cuda_execute_ewald_kernel(kick_params_type **params_ptr, int grid_size);

std::pair<std::function<bool()>, kick_return*> cuda_execute_kick_kernel(kick_params_type *params_ptr, int grid_size,
		cudaStream_t stream);
