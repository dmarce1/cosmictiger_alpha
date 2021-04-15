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


struct group_info_t {
	array<double, NDIM> pos;
	array<float, NDIM> vel;
	int next_id;
	float vtot;
	float epot;
	float ekin;
	int count;
	float rmax;
	float r50;
	float ravg;
	group_info_t() {
		for( int dim = 0; dim < NDIM; dim++) {
			pos[dim] = 0.0;
			vel[dim] = 0.0;
		}
		epot = 0.0;
		ekin = 0.0;
		count = 0;
		rmax = 0.0;
		r50 = 0.0;
		ravg = 0.0;
		vtot = 0.0;
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

	CUDA_EXPORT
	void call_destructors() {
		next_checks.~vector<tree_ptr>();
		checks.~stack_vector<tree_ptr>();
		opened_checks.~vector<tree_ptr>();
	}

	CUDA_EXPORT
	~group_param_type() {

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
