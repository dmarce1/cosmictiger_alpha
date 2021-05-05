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
	static particle_sets* parts;
	static size_t my_start;
	static size_t my_stop;
	static size_t my_size;
	static size_t global_size;
	static int nprocs;
	static int rank;
	static std::vector<hpx::id_type> localities;

	static int index_to_rank(size_t);
public:
	static size_t local_sort(int, size_t, size_t, double, int);
	static void init();
	static void generate_random();
	static size_t sort(int, size_t, size_t, double, int);
	static void swap_particles(int,particle_arc);
	static void execute_swaps(int, std::vector<sort_quantum>);

};

#endif /* PARTICLE_SERVER_HPP_ */
