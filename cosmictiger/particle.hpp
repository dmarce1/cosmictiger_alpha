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

#define NO_GROUP (0x7FFFFFFFFFFF)

using group_t = unsigned long long int;

struct rungbits {
	uint8_t lo : 4;
	uint8_t hi : 4;
};

#include <cosmictiger/array.hpp>
#include <atomic>
#include <vector>

using rung_t = int8_t;

struct particle_set {
	particle_set() = default;
	particle_set(size_t, size_t = 0);
	CUDA_EXPORT
	~particle_set();CUDA_EXPORT
	fixed32 pos_ldg(int, size_t index) const;CUDA_EXPORT
	fixed32 pos(int, size_t index) const;CUDA_EXPORT
	float vel(int dim, size_t index) const;CUDA_EXPORT
	rung_t rung(size_t index) const;
	size_t sort_range(size_t begin, size_t end, double xmid, int xdim);CUDA_EXPORT
	fixed32& pos(int dim, size_t index);CUDA_EXPORT
	float& vel(int dim, size_t index);CUDA_EXPORT
	void set_rung(rung_t t, size_t index);
	void generate_random();
	void load_particles(std::string filename);
	void load_from_file(FILE* fp);
	void save_to_file(FILE* fp);
	void generate_grid();CUDA_EXPORT
	group_t group(size_t) const;CUDA_EXPORT
	group_t& group(size_t);CUDA_EXPORT
	group_t get_last_group(size_t) const;CUDA_EXPORT
	void set_last_group(size_t,group_t);CUDA_EXPORT
	size_t size() const {
		return size_;
	}
	void finish_groups();
	void init_groups();CUDA_EXPORT
	float force(int dim, size_t index) const;CUDA_EXPORT
	float& force(int dim, size_t index);CUDA_EXPORT
	float pot(size_t index) const;CUDA_EXPORT
	float& pot(size_t index);
	void silo_out(const char* filename) const;
	#ifndef __CUDACC__
protected:
#endif
	array<fixed32*, NDIM> xptr_;
	array<float,NDIM>* uptr_;
	rung_t* rptr_;
	group_t* idptr_;
	rungbits* srptr_;
	uint32_t* lidptr1_;
	uint16_t* lidptr2_;
	array<float*, NDIM> gptr_;
	float* eptr_;
	size_t size_;
	size_t offset_;
	bool virtual_;

public:
	CUDA_EXPORT
	particle_set get_virtual_particle_set() const {
		particle_set v;
		v.rptr_ = rptr_;
		v.idptr_ = idptr_;
		v.lidptr1_ = lidptr1_;
		v.lidptr2_ = lidptr2_;
		v.uptr_ = uptr_;
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

	assert(index < size_);
	return xptr_[dim][index];
}

inline float particle_set::vel(int dim, size_t index) const {

	assert(index < size_);
	return uptr_[index][dim];
}

CUDA_EXPORT
inline rung_t particle_set::rung(size_t index) const {

	assert(index < size_);
	/*if (rptr_[index] != uptr_[index].p.r) {
		printf("%i %i\n", rptr_[index], uptr_[index].p.r);
	}
	return uptr_[index].p.r;*/
	return rptr_[index];
}

CUDA_EXPORT
inline fixed32& particle_set::pos(int dim, size_t index) {

	assert(index < size_);
	return xptr_[dim][index];
}

CUDA_EXPORT
inline float& particle_set::vel(int dim, size_t index) {

	assert(index < size_);
	return uptr_[index][dim];
}

CUDA_EXPORT
inline void particle_set::set_rung(rung_t t, size_t index) {

	assert(index < size_);
	rptr_[index] = t;
	//uptr_[index].p.r = t;
}

void drift(particle_set *parts, double a1, double a2, double dtau);

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

CUDA_EXPORT inline group_t particle_set::get_last_group(size_t index) const {
	return lidptr1_[index] | (group_t(lidptr2_[index]) << 32ULL);
}

CUDA_EXPORT inline void particle_set::set_last_group(size_t index, group_t g) {
	lidptr1_[index] = g & 0xFFFFFFFFULL;
	lidptr2_[index] = g >> 32ULL;
}

#endif /* COSMICTIGER_PARTICLE_HPP_ */
