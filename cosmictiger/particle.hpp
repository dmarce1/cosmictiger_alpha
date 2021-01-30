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

#include <array>
#include <atomic>
#include <vector>

using rung_t = int8_t;

struct particle {
   std::array<fixed32, NDIM> x;
   std::array<float, NDIM> v;
   struct flags_t {
      uint64_t morton_id :56;
      uint64_t rung :8;
   };
   flags_t flags;
   template<class A>
   void serialize(A &&arc, unsigned) {
      arc & x;
      arc & v;
      uint64_t tmp1 = flags.rung;
      uint64_t tmp2 = flags.morton_id;
      arc & tmp1;
      arc & tmp2;
      flags.rung = tmp1;
      flags.morton_id = tmp2;
   }
};

struct particle_set {
   enum format {
      soa, aos
   };

   particle_set(size_t, size_t = 0);
   ~particle_set();
   fixed32 pos(int dim, size_t index) const;
   std::array<fixed32, NDIM> pos(size_t index) const;
   float vel(int dim, size_t index) const;
   rung_t rung(size_t index) const;
   morton_t mid(size_t index) const;
   void set_mid(morton_t, size_t index);
   fixed32& pos(int dim, size_t index);
   float& vel(int dim, size_t index);
   void set_rung(rung_t t, size_t index);
   particle part(size_t index) const;
   std::vector<size_t> local_sort(size_t, size_t, int64_t, morton_t key_begin, morton_t key_end );
   void generate_random();
#ifndef __CUDACC__
private:
#endif
   std::array<fixed32*, NDIM> xptr_;
   std::array<float*, NDIM> vptr_;
   particle::flags_t *rptr_;
   particle *pptr_;
   format format_;
   size_t size_;
   size_t offset_;
};

std::vector<size_t> cuda_keygen(particle_set &set, size_t start, size_t stop, int depth, morton_t, morton_t);

inline std::array<fixed32, NDIM> particle_set::pos(size_t index) const {
   std::array<fixed32, NDIM> x;
   for (int dim = 0; dim < NDIM; dim++) {
      x[dim] = pos(dim, index);
   }
   return x;
}

inline fixed32 particle_set::pos(int dim, size_t index) const {
   return xptr_[dim][index - offset_];
}

inline float particle_set::vel(int dim, size_t index) const {
   return vptr_[dim][index - offset_];
}

inline rung_t particle_set::rung(size_t index) const {
   return rptr_[index - offset_].rung;
}

inline morton_t particle_set::mid(size_t index) const {
   return rptr_[index - offset_].morton_id;
}

inline fixed32& particle_set::pos(int dim, size_t index) {
   return xptr_[dim][index - offset_];
}

inline float& particle_set::vel(int dim, size_t index) {
   return vptr_[dim][index - offset_];
}

inline void particle_set::set_mid(morton_t t, size_t index) {
   rptr_[index - offset_].morton_id = t;
}

inline void particle_set::set_rung(rung_t t, size_t index) {
   rptr_[index - offset_].rung = t;
}

inline particle particle_set::part(size_t index) const {
   particle p;
   for (int dim = 0; dim < NDIM; dim++) {
      p.x[dim] = pos(dim, index);
   }
   for (int dim = 0; dim < NDIM; dim++) {
      p.v[dim] = vel(dim, index);
   }
   p.flags.rung = rung(index);
   p.flags.morton_id = mid(index);
   return p;
}

#endif /* COSMICTIGER_PARTICLE_HPP_ */