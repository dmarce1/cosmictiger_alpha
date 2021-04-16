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


struct group_info_t {
	group_t id;
	array<double, NDIM> pos;
	array<float, NDIM> vel;
	std::shared_ptr<std::atomic<int>> count;
	float epot;
	float ekin;
	float rmax;
	float ravg;
	group_info_t() {
		count = std::make_shared<std::atomic<int>>(0);
	}
};

void group_info_add(group_t id, const array<fixed32, NDIM>& pos, const array<float, NDIM>& vel);

struct group_param_type {
	particle_set parts;
	tree_ptr self;
	float link_len;
	vector<tree_ptr> next_checks;
	vector<tree_ptr> tmp;
	vector<tree_ptr> opened_checks;
	stack_vector<tree_ptr> checks;
	bool first_round;
	int depth;
	size_t block_cutoff;



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
hpx::future<void> find_groups_phase1(group_param_type*);
bool find_groups_phase2(group_param_type* params);
#endif
bool call_cuda_find_groups_phase2(group_param_type* params, const vector<tree_ptr>& leaves);

struct groups_shmem {
	array<array<fixed32, GROUP_BUCKET_SIZE>, NDIM> others;
	array<array<fixed32, GROUP_BUCKET_SIZE>, NDIM> self;
};

void group_data_create(particle_set& parts);
void group_data_reduce();
void group_data_output(FILE* fp);
void group_data_destroy();


#endif /* GROUPS_HPP_ */
