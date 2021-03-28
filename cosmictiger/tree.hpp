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

template<class A, class B>
struct pair {
	A first;
	B second;
};

#define LEFT 0
#define RIGHT 1
//#define KICK_CUDA_SIZE (1<<16)

#define TREE_PTR_STACK (TREE_MAX_DEPTH*WORKSPACE_SIZE)

class tree;
struct sort_params;
struct tree_ptr;


#ifndef __CUDACC__
class pranges {
private:
	struct range_less {
		bool operator()(const pair<size_t, size_t>& a, const pair<size_t, size_t>& b) const {
			return a.first < b.first;
		}
	};
	using set_type = std::set<pair<size_t,size_t>, range_less>;
	set_type ranges;
	mutex_type mtx;
public:
	void add_range(pair<size_t, size_t> r) {
		std::lock_guard<mutex_type> lock(mtx);
		ranges.insert(r);
	}
	void clear() {
		ranges.clear();
	}
	bool range_is_covered(pair<size_t, size_t> r) {
		std::lock_guard<mutex_type> lock(mtx);
		std::vector<pair<size_t, size_t>> tmp;
		bool rc = false;
		for (auto i = ranges.begin(); i != ranges.end(); i++) {
			auto this_range = *i;
			auto j = i++;
			while (j != ranges.end() && i->second == j->first) {
				this_range.second = j->second;
				i++;
				j++;
			}
			if (this_range.first <= r.first && this_range.second >= r.second) {
				rc = true;
			}
			tmp.push_back(this_range);
		}
		ranges.clear();
		ranges.insert(tmp.begin(), tmp.end());
		tmp.clear();
		return rc;
	}
};

#endif

struct tree_stats {
	int max_depth;
	int min_depth;
	int e_depth;
	size_t nparts;
	size_t nleaves;
	size_t nnodes;
};

#ifndef __CUDACC__
struct tree_alloc {
	managed_allocator<tree> tree_alloc;
};

struct sort_params {
#ifdef TEST_STACK
	uint8_t* stack_ptr;
#endif
	range box;
	std::shared_ptr<tree_alloc> allocs;
	int8_t depth;
	int8_t min_depth;
	double theta;
	pair<size_t, size_t> parts;
	int min_rung;
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
		parts.first = 0;
		parts.second = global().opts.nparts;
		depth = 0;
		allocs = std::make_shared<tree_alloc>();
	}

	std::array<sort_params, NCHILD> get_children() const {
		std::array<sort_params, NCHILD> child;
		for (int i = 0; i < NCHILD; i++) {
			child[i].depth = depth + 1;
			child[i].allocs = allocs;
			child[i].box = box;
			child[i].min_depth = min_depth;
			child[i].min_rung = min_rung;
		}
		return child;
	}
};

#endif

class tree_ptr;
class kick_params_type;

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

struct sort_return {
	tree_stats stats;
	tree_ptr check;
	size_t active_parts;
	size_t active_nodes;
	template<class A>
	void serialization(A &&arc, unsigned) {
		assert(false);
	}
};
#define NITERS 4
struct cuda_ewald_shmem {
	array<float, KICK_BLOCK_SIZE> Lreduce;  // 256
};

struct multipole_pos {
	multipole multi;
	array<fixed32, NDIM> pos;
};

struct check_data {
	array<fixed32, NDIM> pos;
	float radius;
	bool isleaf;
};

struct cuda_kick_shmem {
	union {
		array<array<fixed32, KICK_PP_MAX>, NDIM> src;  // 3072
		array<multipole_pos, KICK_PC_MAX> msrc;
	};
	array<array<fixed32, MAX_BUCKET_SIZE>, NDIM> sink;  // 3072
	array<uint8_t, MAX_BUCKET_SIZE> act_map;
};
struct list_sizes_t {
	int multi;
	int part;
	int tmp;
	int open;
	int next;
};

list_sizes_t get_list_sizes();
void reset_list_sizes();

struct kick_params_type {
	vector<tree_ptr> multi_interactions;
	vector<tree_ptr> part_interactions;
	vector<tree_ptr> next_checks;
	vector<tree_ptr> opened_checks;
	vector<tree_ptr> tmp;
	stack_vector<tree_ptr> dchecks;
	stack_vector<tree_ptr> echecks;
	array<array<float, MAX_BUCKET_SIZE>, NDIM> F;
	array<float, MAX_BUCKET_SIZE> Phi;
	array<expansion<float>, TREE_MAX_DEPTH> L;
	array<array<fixed32, NDIM>, TREE_MAX_DEPTH> Lpos;
	tree_ptr tptr;
	particle_set* particles;
	int depth;
	size_t block_cutoff;
	float theta;
	float M;
	float G;
	float eta;
	float scale;
	float hsoft;
	bool full_eval;
	bool first;
	int rung;
	bool cpu_block;
	float t0;
	kick_params_type& operator=(kick_params_type& other) {
		first = other.first;
		dchecks = other.dchecks.copy_top();
		echecks = other.echecks.copy_top();
		L[other.depth] = other.L[other.depth];
		Lpos[other.depth] = other.Lpos[other.depth];
		depth = other.depth;
		theta = other.theta;
		eta = other.eta;
		scale = other.scale;
		t0 = other.t0;
		hsoft = other.hsoft;
		rung = other.rung;
		full_eval = other.full_eval;
		block_cutoff = other.block_cutoff;
		G = other.G;
		M = other.M;
		return *this;
	}

	inline kick_params_type() {
		depth = 0;
		theta = 0.4;
		eta = 0.2 / std::sqrt(2);
		scale = 1.0;
		t0 = 1.0;
		cpu_block = false;
		first = true;
		full_eval = false;
		rung = 0;
		hsoft = global().opts.hsoft;
		theta = global().opts.theta;
		M = global().opts.M;
		G = global().opts.G;
		const auto s = get_list_sizes();
		//	printf( "%i %i %i %i %i\n", s.part, s.multi, s.next, s.open, s.tmp);
		part_interactions.reserve(s.part);
		multi_interactions.reserve(s.multi);
		next_checks.reserve(s.next);
		opened_checks.reserve(s.open);
		tmp.reserve(s.tmp);
	}
	friend class tree_ptr;
};

struct kick_params_type;

#ifndef __CUDACC__
struct gpu_kick {
	kick_params_type *params;
	std::shared_ptr<hpx::lcos::local::promise<void>> promise;
	pair<size_t, size_t> parts;
	gpu_kick() {
		promise = std::make_shared<hpx::lcos::local::promise<void>>();
	}
};

struct gpu_ewald {
	kick_params_type *params;
	hpx::lcos::local::promise<int32_t> promise;
};
#endif

struct tree {
#ifndef __CUDACC__
private:
#endif //*** multi and pos MUST be adjacent and ordered multi then pos !!!!!!! *****/
	multipole multi;
	array<fixed32, NDIM> pos;
	float radius;
	size_t active_parts;
	size_t active_nodes;
	array<tree_ptr, NCHILD> children;
	pair<size_t, size_t> parts;
	static particle_set* particles;
public:
	tree() {
		children[LEFT].ptr = 0;
		children[RIGHT].ptr = 0;
	}
	static std::atomic<int> cuda_node_count;
	static std::atomic<int> cpu_node_count;
	static void set_cuda_particle_set(particle_set*);
	static void cuda_set_kick_params(particle_set *p);
	static void show_timings();
	#ifndef __CUDACC__
	static pranges covered_ranges;
	static void set_particle_set(particle_set*);
	inline static hpx::future<sort_return> create_child(sort_params&, bool try_thread);
	static fast_future<sort_return> cleanup_child();
	static hpx::lcos::local::mutex mtx;
	static hpx::lcos::local::mutex gpu_mtx;
	hpx::future<void> send_kick_to_gpu(kick_params_type *params);
	static void gpu_daemon();
	inline bool is_leaf() const {
		return children[0] == tree_ptr();
	}
	static void cleanup();
	void cpu_cc_direct(kick_params_type *params);
	void cpu_cp_direct(kick_params_type *params);
	void cpu_pp_direct(kick_params_type *params);
	void cpu_pc_direct(kick_params_type *params);
	void cpu_cc_ewald(kick_params_type *params);
	sort_return sort(sort_params = sort_params());
	hpx::future<void> kick(kick_params_type*);
	static std::atomic<bool> daemon_running;
	static std::atomic<bool> shutdown_daemon;
	static lockfree_queue<gpu_kick, GPU_QUEUE_SIZE> gpu_queue;
	static lockfree_queue<gpu_ewald, GPU_QUEUE_SIZE> gpu_ewald_queue;
#endif
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

void cuda_execute_kick_kernel(kick_params_type *params_ptr, int grid_size, cudaStream_t stream);

