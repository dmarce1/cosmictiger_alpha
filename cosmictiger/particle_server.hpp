/*
 * particle_server.hpp
 *
 *  Created on: May 17, 2021
 *      Author: dmarce1
 */

#ifndef PARTICLE_SERVER_HPP_
#define PARTICLE_SERVER_HPP_


#include <cosmictiger/particle.hpp>
#include <cosmictiger/defs.hpp>

class particle_server {
	static particle_set* parts;
	static std::vector<part_int> free_indices;
	static particle_send_type part_sends;
	static domain_bounds dbounds;
	static spinlock_type mutex;
public:
	static void init();
	static void domain_decomp_gather();
	static void domain_decomp_send();
	static void domain_decomp_finish();
	static void generate_random();
	static void check_domain_bounds();
	static const domain_bounds& get_domain_bounds();
	static particle_set& get_particle_set();
	static void domain_decomp_transmit(std::vector<particle>);
};


#endif /* PARTICLE_SERVER_HPP_ */
