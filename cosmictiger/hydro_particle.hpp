/*
 * hydro_particle.hpp
 *
 *  Created on: Apr 24, 2021
 *      Author: dmarce1
 */

#ifndef SPH_PARTICLE_HPP_
#define SPH_PARTICLE_HPP_

#include <cosmictiger/particle.hpp>

struct hydro_vars {
	array<float,NDIM> mom;
	float ene;
};

class hydro_particle_set : public particle_set {

public:

	hydro_particle_set() = default;
	hydro_particle_set(size_t, size_t = 0);

	CUDA_EXPORT float& smooth_len(size_t i) {
		assert(i<size());
		return hptr_[i];
	}
	CUDA_EXPORT float& energy(size_t i) {
		assert(i<size());
		return eneptr_[i];
	}
	CUDA_EXPORT float& stored_mom(int dim,size_t i) {
		assert(i<size());
		return store_[i].mom[dim];
	}
	CUDA_EXPORT float& stored_ene(size_t i) {
		assert(i<size());
		return store_[i].ene;
	}

	CUDA_EXPORT float smooth_len(size_t i) const {
		assert(i<size());
		return hptr_[i];
	}
	CUDA_EXPORT float energy(size_t i) const {
		assert(i<size());
		return eneptr_[i];
	}
	CUDA_EXPORT float stored_mom(int dim,size_t i) const {
		assert(i<size());
		return store_[i].mom[dim];
	}
	CUDA_EXPORT float stored_ene(size_t i) const {
		assert(i<size());
		return store_[i].ene;
	}
	CUDA_EXPORT int8_t semiactive(size_t i) const {
		assert(i<size());
		return semi_[i];
	}
	CUDA_EXPORT int8_t& semiactive(size_t i) {
		assert(i<size());
		return semi_[i];
	}

	CUDA_EXPORT
	size_t sort_range(size_t begin, size_t end, double xm, int xdim);


	CUDA_EXPORT
	hydro_particle_set get_virtual_particle_set() const;

private:

	float* hptr_;
	float* eneptr_;
	int8_t* semi_;
	hydro_vars* store_;
};


#endif /* SPH_PARTICLE_HPP_ */
