#pragma once

#include <cosmictiger/defs.hpp>
#include <cosmictiger/particle_sets.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/multipole.hpp>
#include <cosmictiger/fast_future.hpp>
#include <cosmictiger/expansion.hpp>
#include <cosmictiger/lockfree_queue.hpp>
#include <cosmictiger/interactions.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/stack_vector.hpp>
#include <cosmictiger/tree_database.hpp>


#include <functional>

#include <queue>
#include <memory>
#include <stack>

#define LEFT 0
#define RIGHT 1

#define TREE_PTR_STACK (TREE_MAX_DEPTH*WORKSPACE_SIZE)

class tree;
struct sort_params;
struct tree_ptr;

struct tree_stats {
	int max_depth;
	int min_depth;
	int e_depth;
	size_t nparts;
	size_t nleaves;
	size_t nnodes;
};

#ifndef __CUDACC__

struct sort_params {
#ifdef TEST_STACK
	uint8_t* stack_ptr;
#endif
	range box;
	int8_t depth;
	int8_t min_depth;
	double theta;
	array<pair<size_t, size_t>,NPART_TYPES> parts;
	int min_rung;
	tree_ptr tptr;
	bool group_sort;
	particle_sets* part_sets;
	std::shared_ptr<tree_allocator> alloc;
	template<class A>
	void serialization(A &&arc, unsigned) {
		/********* ADD******/
	}

	sort_params() {
		depth = -1;
		group_sort = false;
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
		for( int i = 0; i < NPART_TYPES; i++) {
			parts[i].first = 0;
			parts[i].second = part_sets->sets[i]->size();
		}
	}

	std::array<sort_params, NCHILD> get_children() const {
		std::array<sort_params, NCHILD> child;
		for (int i = 0; i < NCHILD; i++) {
			child[i].depth = depth + 1;
			child[i].box = box;
			child[i].min_depth = min_depth;
			child[i].min_rung = min_rung;
			child[i].alloc = alloc;
			child[i].group_sort = group_sort;
			child[i].part_sets = part_sets;

		}
		return child;
	}
};

#endif

class tree_ptr;
class kick_params_type;

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
	vector<tree_ptr> multi_interactions;
	vector<tree_ptr> part_interactions;
	vector<tree_ptr> next_checks;
	vector<tree_ptr> opened_checks;
	stack_vector<tree_ptr> dchecks;
	tree_ptr self;
	int depth;
	array<uint8_t, MAX_BUCKET_SIZE> act_map;
};
struct list_sizes_t {
	int multi;
	int part;
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
	stack_vector<tree_ptr> dchecks;
	stack_vector<tree_ptr> echecks;
	array<array<float, MAX_BUCKET_SIZE>, NDIM> F;
	array<float, MAX_BUCKET_SIZE> Phi;
	array<expansion<float>, TREE_MAX_DEPTH> L;
	array<array<fixed32, NDIM>, TREE_MAX_DEPTH> Lpos;
	particle_sets* part_sets;
	tree_ptr tptr;
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
	bool groups;
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
		tptr = other.tptr;
		groups = other.groups;
		part_sets = other.part_sets;
		return *this;
	}

	inline kick_params_type() {
		groups = false;
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
public:
	static std::atomic<int> cuda_node_count;
	static std::atomic<int> cpu_node_count;
	static void set_cuda_particle_set(particle_set*);
	static void cuda_set_kick_params(particle_set *p);
	static void show_timings();
#ifndef __CUDACC__
//		static pranges covered_ranges;
	inline static fast_future<sort_return> create_child(sort_params&, bool try_thread);
	static fast_future<sort_return> cleanup_child();
	static hpx::lcos::local::mutex mtx;
	static hpx::lcos::local::mutex gpu_mtx;
	static hpx::future<void> send_kick_to_gpu(kick_params_type *params);
	static void gpu_daemon();
	static void cleanup();
	static void cpu_cc_direct(kick_params_type *params);
	static void cpu_cp_direct(kick_params_type *params);
	static void cpu_pp_direct(kick_params_type *params);
	static void cpu_pc_direct(kick_params_type *params);
	static void cpu_cc_ewald(kick_params_type *params);
	static sort_return sort(sort_params = sort_params());
	static hpx::future<void> kick(kick_params_type*);
	static std::atomic<bool> daemon_running;
	static std::atomic<bool> shutdown_daemon;
	static lockfree_queue<gpu_kick, GPU_QUEUE_SIZE> gpu_queue;
	static lockfree_queue<gpu_ewald, GPU_QUEUE_SIZE> gpu_ewald_queue;
#endif
	friend class tree_ptr;
};

void cuda_execute_kick_kernel(kick_params_type *params_ptr, int grid_size, cudaStream_t stream);

struct kick_constants {
	float theta;
	float eta;
	float scale;
	float t0;
	float h;
	float G;
	float M;
	float theta2;
	float invlog2;
	float GM;
	float tfactor;
	float logt0;
	float ainv;
	float halft0;
	float h2;
	float hinv;
	float th;
	int rung;
	int minrung;
	bool full_eval;
	bool first;
	bool groups;
};

void cuda_set_kick_constants(kick_constants consts,particle_sets&);
