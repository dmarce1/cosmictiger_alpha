/*
 * groups.hpp
 *
 *  Created on: Apr 3, 2021
 *      Author: dmarce1
 */

#ifndef GROUPS_HPP_
#define GROUPS_HPP_

#include <cosmictiger/tree.hpp>


struct group_param_type {
	particle_set parts;
	tree_ptr self;
	float link_len;
	vector<tree_ptr> next_checks;
	vector<tree_ptr> opened_checks;
	stack_vector<tree_ptr> checks;
	bool first_round;
	int depth;
	size_t block_cutoff;

	group_param_type() {
		depth = 0;
	}

	group_param_type& operator=(const group_param_type& other) {
		checks = other.checks.copy_top();
		parts = other.parts;
		self = other.self;
		link_len = other.link_len;
		first_round = other.first_round;
		return *this;
	}
};


#ifndef __CUDACC__
hpx::future<bool> find_groups(group_param_type*);
#endif

__device__ bool cuda_find_groups(group_param_type*);
std::function<std::vector<bool>()> call_cuda_find_groups(group_param_type** params, int, cudaStream_t stream);


#endif /* GROUPS_HPP_ */
