/*
 * particle_server.hpp
 *
 *  Created on: May 17, 2021
 *      Author: dmarce1
 */

#ifndef PARTICLE_SERVER_HPP_
#define PARTICLE_SERVER_HPP_

#include <cosmictiger/particle.hpp>
#include <unordered_set>
#include <cosmictiger/defs.hpp>

class tree_ptr;

#define PARTICLE_CACHE_SIZE 1024
#define PARTICLE_CACHE_LINE_SIZE (8*1024)

struct tree_hash;


class particle_server {
	static particle_set* parts;
	static vector<part_int> free_indices;
	static domain_bounds dbounds;
#ifndef __CUDACC__
	static particle_send_type part_sends;
	static vector<particle> part_recvs;
	static spinlock_type mutex;
	static shared_mutex_type shared_mutex;
#endif
public:
	static void init();
	static bool domain_decomp_gather();
	static void domain_decomp_send();
	static void domain_decomp_finish();
	static void generate_random();
	static void check_domain_bounds();
	static void load_NGenIC();
	static const domain_bounds& get_domain_bounds();
	static particle_set& get_particle_set();
	static void domain_decomp_transmit(vector<particle>);
	static void apply_domain_decomp();
	static std::vector<fixed32> gather_pos(std::vector<part_iters>);
	static void global_to_local(std::unordered_set<tree_ptr,tree_hash>);
};

#endif /* PARTICLE_SERVER_HPP_ */
