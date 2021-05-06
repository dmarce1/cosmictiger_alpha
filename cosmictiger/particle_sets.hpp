/*
 * particle_sets.hpp
 *
 *  Created on: Apr 24, 2021
 *      Author: dmarce1
 */

#ifndef PARTICLE_SETS_HPP_
#define PARTICLE_SETS_HPP_

#include <cosmictiger/hydro_particle.hpp>
#include <cosmictiger/global.hpp>

#define NPART_TYPES 2
#define CDM_SET 0
#define BARY_SET 1

struct particle_sets {
	array<particle_set*, NPART_TYPES> sets;
	particle_set cdm;
	hydro_particle_set baryon;
	array<float, NPART_TYPES> weights;

	particle_sets();
	void load_from_file(FILE* fp);
	void save_to_file(FILE* fp);
	void generate_random();
	particle_sets(const particle_sets& other);
	particle_sets& operator=(const particle_sets& other);
	particle_sets(size_t size, size_t start = 0);
	particle_sets get_virtual_particle_sets();
	size_t size() const;
};

using part_iters = pair<size_t,size_t>;

struct parts_type: public array<part_iters, NPART_TYPES> {
	CUDA_EXPORT
	size_t index_offset(int index) const {
		size_t sz = 0;
		const array<part_iters, NPART_TYPES>& parts = *this;
		for (int pi = 0; pi < index; pi++) {
			sz += parts[pi].second - parts[pi].first;
//			printf( "%i %i\n",  parts[pi].second, parts[pi].first);
		}
		return sz;
	}
	CUDA_EXPORT
	size_t size() const {
		return index_offset(NPART_TYPES);
	}
};

void silo_out(particle_sets partsets, const char* filename);

#endif /* PARTICLE_SETS_HPP_ */
