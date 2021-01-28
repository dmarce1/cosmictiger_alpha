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

using rung_t = int32_t;

struct particle {
   std::array<fixed32, NDIM> x;
   std::array<float, NDIM> v;
   rung_t rung;
   template<class A>
   void serialize(A &&arc, unsigned) {
      arc & x;
      arc & v;
      arc & rung;
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
   fixed32& pos(int dim, size_t index);
   float& vel(int dim, size_t index);
   rung_t& rung(size_t index);
   particle part(size_t index) const;
   std::vector<size_t> local_sort(size_t, size_t, int64_t);
   void generate_random();
private:
   std::array<fixed32*, NDIM> xptr_;
   std::array<float*, NDIM> vptr_;
   rung_t *rptr_;
   particle *pptr_;
   format format_;
   size_t size_;
   size_t offset_;
};

inline std::array<fixed32, NDIM> particle_set::pos(size_t index) const {
   std::array<fixed32,NDIM> x;
   for( int dim = 0; dim < NDIM; dim++) {
      x[dim] = pos(dim,index);
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
   return rptr_[index - offset_];
}

inline fixed32& particle_set::pos(int dim, size_t index) {
   return xptr_[dim][index - offset_];
}

inline float& particle_set::vel(int dim, size_t index) {
   return vptr_[dim][index - offset_];
}

inline rung_t& particle_set::rung(size_t index) {
   return rptr_[index - offset_];
}

inline particle particle_set::part(size_t index) const {
   particle p;
   for (int dim = 0; dim < NDIM; dim++) {
      p.x[dim] = pos(dim, index);
   }
   for (int dim = 0; dim < NDIM; dim++) {
      p.v[dim] = vel(dim, index);
   }
   p.rung = rung(index);
   return p;
}

#endif /* COSMICTIGER_PARTICLE_HPP_ */
