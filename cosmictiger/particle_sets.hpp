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

	particle_sets() {
		sets[0] = &cdm;
		sets[1] = &baryon;
	}

	void load_from_file(FILE* fp) {
		for( int pi = 0; pi < NPART_TYPES; pi++) {
			sets[pi]->load_from_file(fp);
		}
	}


	void save_to_file(FILE* fp) {
		for( int pi = 0; pi < NPART_TYPES; pi++) {
			sets[pi]->save_to_file(fp);
		}
	}

	void generate_random() {
		const float h = std::pow(baryon.size(),-1.0/3.0);
		cdm.generate_random();
		baryon.generate_random();
		for( int i = 0; i < baryon.size(); i++) {
			 baryon.energy(i) = 0.0;
			 baryon.smooth_len(i) = h;
		}
	}

	particle_sets(const particle_sets& other) {
		sets[0] = &cdm;
		sets[1] = &baryon;
		*this = other;
	}
	particle_sets& operator=(const particle_sets& other) {
		cdm = other.cdm;
		baryon = other.baryon;
		weights = other.weights;
		return *this;
	}

	particle_sets(size_t size) :
			cdm(size), baryon(global().opts.sph ? size : 0) {
		const float omega_b = global().opts.omega_b;
		const float omega_c = global().opts.omega_c;
		sets[0] = &cdm;
		sets[1] = &baryon;
		if (global().opts.sph) {
			weights[0] = omega_c / (omega_b + omega_c);
			weights[1] = omega_b / (omega_b + omega_c);
		} else {
			weights[0] = 1.0;
			weights[1] = 0.0;
		}
	}

	particle_sets get_virtual_particle_sets() {
		particle_sets v;
		v.cdm = cdm.get_virtual_particle_set();
		v.baryon = baryon.get_virtual_particle_set();
		v.weights = weights;
		return v;
	}

	size_t size() const {
		size_t sz = 0;
		for (int i = 0; i < NPART_TYPES; i++) {
			sz += sets[i]->size();
		}
		return sz;
	}
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

#endif /* PARTICLE_SETS_HPP_ */
