/*
 * particle.hpp
 *
 *  Created on: Jan 23, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_PARTICLE_HPP_
#define COSMICTIGER_PARTICLE_HPP_

#include <cosmictiger/defs.hpp>
#include <cosmictiger/memory.hpp>
#include <cosmictiger/fixed.hpp>
#include <cosmictiger/range.hpp>
#include <cosmictiger/array.hpp>
#ifdef _CUDA_ARCH_
#include <cosmictiger/hpx.hpp>
#endif

struct range;

union pos_type {
	struct {
		fixed32 x;
		fixed32 y;
		fixed32 z;
	} p;
	array<fixed32, NDIM> a;CUDA_EXPORT
	pos_type() {
	}
};

union vel_type {
	struct {
		float x;
		float y;
		float z;
	} p;
	array<fixed32, NDIM> a;CUDA_EXPORT
	vel_type() {
	}
};

#include <cosmictiger/array.hpp>
#include <atomic>
#include <vector>

using rung_t = int8_t;

struct particle_set {
	particle_set() = default;
	particle_set(size_t, size_t = 0);
	void prepare_sort();
	void prepare_kick();
	void prepare_drift();
	~particle_set();CUDA_EXPORT
	pos_type pos(size_t index) const;
	vel_type vel(size_t index) const;CUDA_EXPORT
	rung_t rung(size_t index) const;
	size_t sort_range(size_t begin, size_t end, double xmid, int xdim);CUDA_EXPORT
	pos_type& pos(size_t index);CUDA_EXPORT
	vel_type& vel(size_t index);CUDA_EXPORT
	void set_rung(rung_t t, size_t index);
	void generate_random();
	void load_particles(std::string filename);
	void generate_grid();CUDA_EXPORT
	size_t size() const {
		return size_;
	}
#ifdef TEST_FORCE
	CUDA_EXPORT
	float force(int dim, size_t index) const;CUDA_EXPORT
	float& force(int dim, size_t index);CUDA_EXPORT
	float pot(size_t index) const;CUDA_EXPORT
	float& pot(size_t index);
#endif
#ifndef __CUDACC__
private:
#endif
	pos_type* pptr_;
	vel_type* uptr_;
#ifdef TEST_FORCE
	array<float*, NDIM> gptr_;
	float* eptr_;
#endif
	rung_t *rptr_;
	void *base_;
	size_t size_;
	size_t offset_;
	bool virtual_;

public:

	particle_set get_virtual_particle_set() const {
		particle_set v;
		v.uptr_ = uptr_;
		v.pptr_ = pptr_;
		v.rptr_ = rptr_;
		v.base_ = base_;
#ifdef TEST_FORCE
		v.gptr_ = gptr_;
		v.eptr_ = eptr_;
#endif
		v.size_ = size_;
		v.offset_ = offset_;
		v.virtual_ = true;
		return v;
	}
}
;

CUDA_EXPORT
inline pos_type particle_set::pos(size_t index) const {
	assert(index >= 0);
	assert(index < size_);
	return pptr_[index - offset_];
}

inline vel_type particle_set::vel(size_t index) const {
	assert(index >= 0);
	assert(index < size_);
	return uptr_[index - offset_];
}

CUDA_EXPORT
inline rung_t particle_set::rung(size_t index) const {
	assert(index >= 0);
	assert(index < size_);
	return rptr_[index - offset_];
}
CUDA_EXPORT
inline pos_type& particle_set::pos(size_t index) {
	assert(index >= 0);
	assert(index < size_);
	return pptr_[index - offset_];
}

CUDA_EXPORT
inline vel_type& particle_set::vel(size_t index) {
	assert(index >= 0);
	assert(index < size_);
	return uptr_[index - offset_];
}

CUDA_EXPORT
inline void particle_set::set_rung(rung_t t, size_t index) {
	assert(index >= 0);
	assert(index < size_);
	rptr_[index - offset_] = t;
}

void drift(particle_set *parts, double a1, double a2, double dtau);

#ifdef TEST_FORCE
CUDA_EXPORT inline float particle_set::force(int dim, size_t index) const {
	assert(index < size_);
	return gptr_[dim][index - offset_];

}
CUDA_EXPORT inline float& particle_set::force(int dim, size_t index) {
	assert(index < size_);
	return gptr_[dim][index - offset_];

}
CUDA_EXPORT inline float particle_set::pot(size_t index) const {
	assert(index < size_);
	return eptr_[index - offset_];

}
CUDA_EXPORT inline float& particle_set::pot(size_t index) {
	assert(index < size_);
	return eptr_[index - offset_];

}
#endif

#endif /* COSMICTIGER_PARTICLE_HPP_ */
