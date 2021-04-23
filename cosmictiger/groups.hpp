/*
 * groups.hpp
 *
 *  Created on: Apr 3, 2021
 *      Author: dmarce1
 */

#ifndef GROUPS_HPP_
#define GROUPS_HPP_

#include <cosmictiger/tree.hpp>
#include <cosmictiger/particle.hpp>
#include <cosmictiger/hpx.hpp>

#include <map>

template<typename T>
class my_allocator: public std::allocator<T> {
public:
	typedef size_t size_type;
	typedef T* pointer;
	typedef const T* const_pointer;
	typedef T value_type;

	template<typename _Tp1>
	struct rebind {
		typedef my_allocator<_Tp1> other;
	};

	pointer allocate(size_type n, const void *hint = 0) {
		unified_allocator alloc;
		return (pointer) alloc.allocate(n * sizeof(T));
	}

	void deallocate(pointer p, size_type n) {
		unified_allocator alloc;
		alloc.deallocate(p);
	}

	my_allocator() throw () :
			std::allocator<T>() {
	}
	my_allocator(const my_allocator &a) throw () :
			std::allocator<T>(a) {
	}
	template<class U>
	my_allocator(const my_allocator<U> &a) throw () :
			std::allocator<T>(a) {
	}
	~my_allocator() throw () {
	}
};

template<class T>
struct atomic_wrapper: public std::atomic<T> {
	atomic_wrapper(atomic_wrapper&& other) {
		auto& ref = (std::atomic<T>&)(*this);
		ref = (T) other;
	}
	atomic_wrapper(const atomic_wrapper& other) {
		auto& ref = (std::atomic<T>&)(*this);
		ref = (T) other;
	}
	atomic_wrapper& operator=(atomic_wrapper&& other) {
		auto& ref = (std::atomic<T>&)(*this);
		ref = (T) other;
		return *this;
	}
	atomic_wrapper() {
	}
	atomic_wrapper& operator=(T value) {
		auto& ref = (std::atomic<T>&)(*this);
		ref = value;
		return *this;
	}
};

struct group_info_t {
	group_t id;
	array<double, NDIM> pos;
	array<float, NDIM> vel;
	array<float, NDIM> vdisp;
	int count;
	float epot;
	float ekin;
	float rmax;
	float r2;
	float ravg;
	float reff;
	float qxx, qxy, qxz, qyy, qyz, qzz;
	float sx, sy, sz;
	std::vector<float, my_allocator<float>> radii;
	std::map<group_t, int, std::less<group_t>, my_allocator<std::pair<const group_t, int>>> parents;
	atomic_wrapper<int> mtx;
	group_info_t() {
		mtx = 0;
	}
	void lock() {
		while ((mtx)++ != 0) {
			(mtx)--;
		}
	}
	void unlock() {
		(mtx)--;
	}
};

struct group_list_sizes {
	int opened;
	int next;
};

group_list_sizes get_group_list_sizes();

using bucket_t = vector<group_info_t>;

void group_info_add(group_t id, const array<fixed32, NDIM>& pos, const array<float, NDIM>& vel);

struct group_param_type {
	vector<tree_ptr> next_checks;
	vector<tree_ptr> opened_checks;
	stack_vector<tree_ptr> checks;
	particle_set parts;
	tree_ptr self;
	float link_len;
	bool first_round;
	int depth;
	size_t block_cutoff;

	CUDA_EXPORT
	void call_destructors() {
		next_checks.~vector<tree_ptr>();
		opened_checks.~vector<tree_ptr>();
		checks.~stack_vector<tree_ptr>();
	}

	group_param_type& operator=(const group_param_type& other) {
		checks = other.checks.copy_top();
		parts = other.parts;
		self = other.self;
		link_len = other.link_len;
		first_round = other.first_round;
		depth = other.depth;
		block_cutoff = other.block_cutoff;
		return *this;
	}
};

#ifndef __CUDACC__
hpx::future<size_t> find_groups(group_param_type* params_ptr);
#endif
std::function<std::vector<size_t>()> call_cuda_find_groups(group_param_type** params, int params_size,
		cudaStream_t stream);

struct groups_shmem {
	array<array<fixed32, GROUP_BUCKET_SIZE>, NDIM> others;
	array<array<fixed32, GROUP_BUCKET_SIZE>, NDIM> self_parts;
	array<int, GROUP_BUCKET_SIZE> other_indexes;
	vector<tree_ptr> next_checks;
	vector<tree_ptr> opened_checks;
	stack_vector<tree_ptr> checks;
	particle_set parts;
	float link_len;
	int depth;

};

#ifndef __CUDACC__
hpx::future<void> group_data_create(particle_set& parts);
#endif
void group_data_save(double scale, int filename);
void group_data_output(FILE* fp);
void group_data_destroy();
vector<bucket_t>& group_table();
int& group_table_size();

__device__
void gpu_groups_kick_update(group_t id, float phi);

void cpu_groups_kick_update(group_t id, float phi);

#endif /* GROUPS_HPP_ */
