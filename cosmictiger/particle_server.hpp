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

#define PARTICLE_CACHE_SIZE 1024
#define PARTICLE_CACHE_LINE_SIZE (8*1024)

using pos_data_t = std::vector<std::array<fixed32,NDIM>>;

struct pos_cache_entry {
	pos_data_t X;
	hpx::shared_future<void> ready_fut;
};

struct global_part_iter {
	int rank;
	part_int index;
	bool operator==(const global_part_iter& other) const {
		return rank == other.rank && index == other.index;
	}
};

struct global_part_iter_hash_lo {
	size_t operator()(const global_part_iter& i) const {
		const auto j = i.index / PARTICLE_CACHE_LINE_SIZE * hpx_size() + i.rank;
		return (j) % PARTICLE_CACHE_SIZE;
	}
};

struct global_part_iter_hash_hi {
	size_t operator()(const global_part_iter& i) const {
		const auto j = i.index / PARTICLE_CACHE_LINE_SIZE * hpx_size() + i.rank;
		return (j) / PARTICLE_CACHE_SIZE;
	}
};

using particle_cache_type = std::unordered_map<global_part_iter, std::shared_ptr<pos_cache_entry>, global_part_iter_hash_hi>;

class particle_server {
	static particle_set* parts;
	static std::vector<part_int> free_indices;
	static particle_send_type part_sends;
	static domain_bounds dbounds;
	static spinlock_type mutex;
	static std::array<mutex_type, PARTICLE_CACHE_SIZE> mutexes;
	static std::array<particle_cache_type,PARTICLE_CACHE_SIZE> caches;

	static void load_cache_line(global_part_iter);
public:
	static pos_data_t fetch_cache_line(part_int);
	static void init();
	static bool domain_decomp_gather();
	static void domain_decomp_send();
	static void domain_decomp_finish();
	static void generate_random();
	static void check_domain_bounds();
	static const domain_bounds& get_domain_bounds();
	static particle_set& get_particle_set();
	static void read_positions(std::array<std::vector<fixed32>, NDIM>& x, int rank, part_iters);
	static void domain_decomp_transmit(std::vector<particle>);
	static void apply_domain_decomp();
	static void free_cache();
};

#endif /* PARTICLE_SERVER_HPP_ */
