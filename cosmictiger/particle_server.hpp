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


using pos_line_type = std::array<std::vector<fixed32>,NDIM>;

struct pos_cache_entry {
#ifndef __CUDACC__
	std::shared_ptr<pos_line_type> data;
	std::shared_ptr<hpx::shared_future<void>> fut;
#endif
};

struct part_hash_hi {
	size_t operator()(size_t i) const;
};

#define POS_CACHE_SIZE 1024



using pos_cache_type = std::unordered_map<size_t, pos_cache_entry, part_hash_hi>;


class particle_server {
#ifndef __CUDACC__
	static std::array<std::array<pos_cache_type, POS_CACHE_SIZE>, NPART_TYPES> pos_caches;
	static std::array<std::array<spinlock_type, POS_CACHE_SIZE>, NPART_TYPES> mutexes;
	static std::vector<hpx::id_type> localities;
#endif
	static particle_sets* parts;
	static size_t my_start;
	static size_t my_stop;
	static size_t my_size;
	static size_t global_size;
	static int nprocs;
	static int rank;
	static size_t pos_cache_line_size;
public:
	static int index_to_rank(size_t);
	static particle_sets& get_particle_sets();
	static size_t local_sort(int, size_t, size_t, double, int);
	static void init();
	static pos_line_type read_pos_cache_line(int pi, size_t i);
	static void generate_random();
	static size_t sort(int, size_t, size_t, double, int);
	static particle_arc swap_particles(int, particle_arc);
	static void execute_swaps(int, std::vector<sort_quantum>);
	static fixed32 pos_cache_read(int pi, int dim, size_t i);
	static size_t pos_cache_line_index(size_t i);
	static std::vector<rung_t> read_rungs(int pi, size_t, size_t);
	inline static fixed32 pos_read(int pi, int dim, size_t i) {
		if (i >= my_start & i < my_stop) {
			return parts->sets[pi]->pos(dim, i);
		} else {
			return pos_cache_read(pi, dim, i);
		}
	}
	friend class part_hash_hi;
	friend class part_hash_lo;

};



struct part_hash_lo {
	size_t operator()(size_t i) const {
		return i % particle_server::pos_cache_line_size;
	}
};

inline size_t part_hash_hi::operator()(size_t i) const {
	return i / particle_server::pos_cache_line_size;
}

#endif /* PARTICLE_SERVER_HPP_ */
