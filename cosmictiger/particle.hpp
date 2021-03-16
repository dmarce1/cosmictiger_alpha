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
#ifdef _CUDA_ARCH_
#include <cosmictiger/hpx.hpp>
#endif

struct range;

#include <cosmictiger/array.hpp>
#include <atomic>
#include <vector>

using rung_t = int8_t;

struct particle {
	std::array<fixed32, NDIM> x;
	std::array<float, NDIM> v;
	uint64_t morton_id;
	int8_t rung;
};

struct particle_set {
	particle_set() = default;
/*	CUDA_EXPORT
	inline void swap(size_t a, size_t b) {
		for (int dim = 0; dim < NDIM; dim++) {
			auto tmp1 = xptr_[dim][a];
			auto tmp2 = vptr_[dim][a];
			xptr_[dim][a] = xptr_[dim][b];
			vptr_[dim][a] = vptr_[dim][b];
			xptr_[dim][b] = tmp1;
			vptr_[dim][b] = tmp2;
		}
		auto tmp3 = rptr_[a];
		rptr_[a] = rptr_[b];
		rptr_[b] = tmp3;
	}*/
	particle_set(size_t, size_t = 0);
	void prepare_sort();
	void prepare_kick(size_t begin, size_t end, cudaStream_t stream);
	void prepare_drift(cudaStream_t stream);
	~particle_set();CUDA_EXPORT
	fixed32 pos(int dim, size_t index) const;
	std::array<fixed32, NDIM> pos(size_t index) const;
	float vel(int dim, size_t index) const;CUDA_EXPORT
	rung_t rung(size_t index) const;
	morton_t mid(size_t index) const;
	void set_mid(morton_t, size_t index);CUDA_EXPORT
	size_t sort_range(size_t begin, size_t end, double xmid, int xdim);
	CUDA_EXPORT
	fixed32& pos(int dim, size_t index);CUDA_EXPORT
	float& vel(int dim, size_t index);CUDA_EXPORT
	void set_rung(rung_t t, size_t index);
	particle part(size_t index) const;
	std::vector<size_t> local_sort(size_t, size_t, int64_t, morton_t key_begin, morton_t key_end);
	void generate_random();
	void generate_grid();CUDA_EXPORT
	size_t size() const {
		return size_;
	}
	void load_particles(std::string filename);
#ifdef TEST_FORCE
	CUDA_EXPORT
	float force(int dim, size_t index) const;
	CUDA_EXPORT
	float& force(int dim, size_t index);
	CUDA_EXPORT
	float pot(size_t index) const;
	CUDA_EXPORT
	float& pot(size_t index);
#endif
#ifndef __CUDACC__
private:
#endif
	array<fixed32*, NDIM> xptr_;
	array<float*, NDIM> vptr_;
#ifdef TEST_FORCE
	array<float*, NDIM> gptr_;
	float* eptr_;
#endif
	int8_t *rptr_;
	uint64_t *mptr_;
	void *pptr_;
	size_t size_;
	size_t offset_;
	bool virtual_;

public:

	particle_set get_virtual_particle_set() const {
		particle_set v;
		for (int dim = 0; dim < NDIM; dim++) {
			v.vptr_[dim] = vptr_[dim];
			v.xptr_[dim] = xptr_[dim];
		}
		v.rptr_ = rptr_;
		v.pptr_ = pptr_;
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

std::vector<size_t> cuda_keygen(particle_set &set, size_t start, size_t stop, int depth, morton_t, morton_t,
		cudaStream_t);

inline std::array<fixed32, NDIM> particle_set::pos(size_t index) const {
	assert(index >= 0);
	assert(index < size_);
	std::array<fixed32, NDIM> x;
	for (int dim = 0; dim < NDIM; dim++) {
		x[dim] = pos(dim, index);
	}
	return x;
}

CUDA_EXPORT
inline fixed32 particle_set::pos(int dim, size_t index) const {
	assert(index >= 0);
	assert(index < size_);
	return xptr_[dim][index - offset_];
}

inline float particle_set::vel(int dim, size_t index) const {
	assert(index >= 0);
	assert(index < size_);
	return vptr_[dim][index - offset_];
}

CUDA_EXPORT
inline rung_t particle_set::rung(size_t index) const {
	assert(index >= 0);
	assert(index < size_);
	return rptr_[index - offset_];
}

inline morton_t particle_set::mid(size_t index) const {
	assert(index >= 0);
	assert(index < size_);
	return mptr_[index - offset_];
}
CUDA_EXPORT
inline fixed32& particle_set::pos(int dim, size_t index) {
	assert(index >= 0);
	assert(index < size_);
	return xptr_[dim][index - offset_];
}

CUDA_EXPORT
inline float& particle_set::vel(int dim, size_t index) {
	assert(index >= 0);
	assert(index < size_);
	return vptr_[dim][index - offset_];
}

inline void particle_set::set_mid(morton_t t, size_t index) {
	assert(index >= 0);
	assert(index < size_);
	mptr_[index - offset_] = t;
}

CUDA_EXPORT
inline void particle_set::set_rung(rung_t t, size_t index) {
	assert(index >= 0);
	assert(index < size_);
	rptr_[index - offset_] = t;
}

inline particle particle_set::part(size_t index) const {
	assert(index >= 0);
	assert(index < size_);
	particle p;
	for (int dim = 0; dim < NDIM; dim++) {
		p.x[dim] = pos(dim, index);
	}
	for (int dim = 0; dim < NDIM; dim++) {
		p.v[dim] = vel(dim, index);
	}
	p.rung = rung(index);
	p.morton_id = mid(index);
	return p;
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
