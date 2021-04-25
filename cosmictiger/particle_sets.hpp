/*
 * particle_sets.hpp
 *
 *  Created on: Apr 24, 2021
 *      Author: dmarce1
 */

#ifndef PARTICLE_SETS_HPP_
#define PARTICLE_SETS_HPP_

#include <cosmictiger/hydro_particle.hpp>

#define NPART_TYPES 2

struct particle_sets {
	array<particle_set*, NPART_TYPES> sets;
	particle_set cdm;
	hydro_particle_set baryon;

	particle_sets() {
		sets[0] = &cdm;
		sets[1] = &baryon;
	}

	particle_sets(size_t cdm_size, size_t baryon_size) : cdm(cdm_size), baryon(baryon_size) {
		sets[0] = &cdm;
		sets[1] = &baryon;
	}

	particle_sets get_virtual_particle_sets() {
		particle_sets v;
		v.cdm = cdm.get_virtual_particle_set();
		v.baryon = baryon.get_virtual_particle_set();
		return v;
	}

	size_t size() const {
		size_t sz = 0;
		for( int i = 0; i < NPART_TYPES; i++) {
			sz += sets[i]->size();
		}
		return sz;
	}
};

using part_iters = pair<size_t,size_t>;

struct parts_type : public array<part_iters,NPART_TYPES> {
	size_t index_offset(int index) const {
		size_t sz = 0;
		const array<part_iters,NPART_TYPES>& parts = *this;
		for( int pi = 0; pi < index; pi++) {
			sz += parts[pi].second - parts[pi].first;
		}
		return sz;
	}
	size_t size() const {
		return index_offset(NPART_TYPES);
	}
};

#endif /* PARTICLE_SETS_HPP_ */
