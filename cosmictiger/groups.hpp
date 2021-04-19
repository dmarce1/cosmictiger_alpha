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

struct group_info_t {
	group_t id;
	array<double, NDIM> pos;
	array<float, NDIM> vel;
	array<float, NDIM> vdisp;
	int count;
	float epot;
	float ekin;
	float rmax;
	float ravg;
	float reff;
	std::vector<float> radii;
	std::map<group_t, std::shared_ptr<int>> parents;
	std::shared_ptr<std::atomic<int>> mtx;
	group_info_t() {
		mtx = std::make_shared<std::atomic<int>>(0);
	}
	void lock() {
		while ((*mtx)++ != 0) {
			(*mtx)--;
		}
	}
	void unlock() {
		(*mtx)--;
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
hpx::future<bool> find_groups(group_param_type* params_ptr);
#endif
std::function<std::vector<bool>()> call_cuda_find_groups(group_param_type** params, int params_size,
		cudaStream_t stream);

struct groups_shmem {
	array<array<fixed32, GROUP_BUCKET_SIZE>, NDIM> others;
	array<array<fixed32, GROUP_BUCKET_SIZE>, NDIM> self_parts;
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
