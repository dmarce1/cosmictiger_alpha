/*
 * particle_server.hpp
 *
 *  Created on: May 3, 2021
 *      Author: dmarce1
 */

#ifndef PARTICLE_SERVER_HPP_
#define PARTICLE_SERVER_HPP_



#include <cosmictiger/hpx.hpp>
#include <cosmictiger/particle_sets.hpp>

struct sort_quantum {
	int rank_from;
	int rank_to;
	part_iters range_from;
	part_iters range_to;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & rank_from;
		arc & rank_to;
		arc & range_from;
		arc & range_to;
	}
};

class particle_server {

public:
	static int index_to_rank(size_t);
	static particle_sets& get_particle_sets();
	static size_t local_sort(int, size_t, size_t, double, int);
	static void init();
	static void generate_random();
	static size_t sort(int, size_t, size_t, double, int);
	static void swap_particles(int,particle_arc);
	static void execute_swaps(int, std::vector<sort_quantum>);

};

#endif /* PARTICLE_SERVER_HPP_ */
