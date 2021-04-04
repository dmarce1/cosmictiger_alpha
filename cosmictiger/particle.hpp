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

using group_t = int64_t;

union vel_type {
	struct {
		float x;
		float y;
		float z;
		int8_t r;
	} p;
	array<float, NDIM> a;CUDA_EXPORT
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
	fixed32 pos_ldg(int, size_t index) const;CUDA_EXPORT
	fixed32 pos(int, size_t index) const;CUDA_EXPORT
	vel_type vel(size_t index) const;CUDA_EXPORT
	rung_t rung(size_t index) const;
	size_t sort_range(size_t begin, size_t end, double xmid, int xdim);CUDA_EXPORT
	fixed32& pos(int dim, size_t index);CUDA_EXPORT
	vel_type& vel(size_t index);CUDA_EXPORT
	void set_rung(rung_t t, size_t index);
	void generate_random();
	void load_particles(std::string filename);
	void load_from_file(FILE* fp);
	void save_to_file(FILE* fp);
	void generate_grid();
	CUDA_EXPORT group_t group(size_t) const;
	CUDA_EXPORT group_t& group(size_t);
	CUDA_EXPORT
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
	array<fixed32*, NDIM> xptr_;
	vel_type* uptr_;
	group_t* idptr_;
#ifdef TEST_FORCE
	array<float*, NDIM> gptr_;
	float* eptr_;
#endif
//	rung_t *rptr_;
	size_t size_;
	size_t offset_;
	bool virtual_;

public:

	particle_set get_virtual_particle_set() const {
		particle_set v;
		v.uptr_ = uptr_;
		v.idptr_ = idptr_;
		for (int dim = 0; dim < NDIM; dim++) {
			v.xptr_[dim] = xptr_[dim];
		}
//		v.rptr_ = rptr_;
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
inline fixed32 particle_set::pos_ldg(int dim, size_t index) const {
	assert(index >= 0);
	assert(index < size_);
	union fixed_union {
		fixed32 f;
		int i;
	};
	fixed_union x;
	x.i = LDG((int* )(xptr_[dim] + index));
	return x.f;
}

CUDA_EXPORT
inline fixed32 particle_set::pos(int dim, size_t index) const {
	assert(index >= 0);
	assert(index < size_);
	return xptr_[dim][index];
}

inline vel_type particle_set::vel(size_t index) const {
	assert(index >= 0);
	assert(index < size_);
	return uptr_[index];
}

CUDA_EXPORT
inline rung_t particle_set::rung(size_t index) const {
	assert(index >= 0);
	assert(index < size_);
	return uptr_[index].p.r;
}

CUDA_EXPORT
inline fixed32& particle_set::pos(int dim, size_t index) {
	assert(index >= 0);
	assert(index < size_);
	return xptr_[dim][index];
}

CUDA_EXPORT
inline vel_type& particle_set::vel(size_t index) {
	assert(index >= 0);
	assert(index < size_);
	return uptr_[index];
}

CUDA_EXPORT
inline void particle_set::set_rung(rung_t t, size_t index) {
	assert(index >= 0);
	assert(index < size_);
	uptr_[index].p.r = t;
}

void drift(particle_set *parts, double a1, double a2, double dtau);

#ifdef TEST_FORCE
CUDA_EXPORT inline float particle_set::force(int dim, size_t index) const {
	assert(index < size_);
	return gptr_[dim][index];

}
CUDA_EXPORT inline float& particle_set::force(int dim, size_t index) {
	assert(index < size_);
	return gptr_[dim][index];

}
CUDA_EXPORT inline float particle_set::pot(size_t index) const {
	assert(index < size_);
	return eptr_[index];

}
CUDA_EXPORT inline float& particle_set::pot(size_t index) {
	assert(index < size_);
	return eptr_[index];

}

CUDA_EXPORT inline group_t particle_set::group(size_t index) const {
	return idptr_[index];
}

CUDA_EXPORT inline group_t& particle_set::group(size_t index) {
	return idptr_[index];
}


#endif

#endif /* COSMICTIGER_PARTICLE_HPP_ */
