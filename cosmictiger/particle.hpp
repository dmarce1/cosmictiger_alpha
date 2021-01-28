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
   void set_mem_format(format);

   struct soa_interface {
      soa_interface(particle_set &parts);
      fixed32 pos(int dim, size_t index) const;
      float vel(int dim, size_t index) const;
      rung_t rung(size_t index) const;
      fixed32& pos(int dim, size_t index);
      float& vel(int dim, size_t index);
      rung_t& rung(size_t index);
      particle part(size_t index) const;
   private:
      particle_set &parts_;
   };

   struct aos_interface {
      aos_interface(particle_set &parts);
   private:
      fixed32 pos(int dim, size_t index) const;
      float vel(int dim, size_t index) const;
      rung_t rung(size_t index) const;
      fixed32& pos(int dim, size_t index);
      float& vel(int dim, size_t index);
      rung_t& rung(size_t index);
      particle part(size_t index) const;
      particle& part(size_t index);
   private:
      particle_set &parts_;
   };


private:
   std::array<fixed32*, NDIM> xptr_;
   std::array<float*, NDIM> vptr_;
   rung_t *rptr_;
   particle *pptr_;
   format format_;
   size_t size_;
   size_t offset_;
};

inline particle_set::aos_interface::aos_interface(particle_set &parts) :
      parts_(parts) {
   assert(parts_.format_ == format::aos);

}
inline fixed32 particle_set::aos_interface::pos(int dim, size_t index) const {
   return parts_.pptr_[index - parts_.offset_].x[dim];
}

inline float particle_set::aos_interface::vel(int dim, size_t index) const {
   return parts_.pptr_[index - parts_.offset_].v[dim];
}

inline rung_t particle_set::aos_interface::rung(size_t index) const {
   return parts_.pptr_[index - parts_.offset_].rung;
}

inline fixed32& particle_set::aos_interface::pos(int dim, size_t index) {
   return parts_.pptr_[index - parts_.offset_].x[dim];
}

inline float& particle_set::aos_interface::vel(int dim, size_t index) {
   return parts_.pptr_[index - parts_.offset_].v[dim];
}

inline rung_t& particle_set::aos_interface::rung(size_t index) {
   return parts_.pptr_[index - parts_.offset_].rung;
}

inline particle particle_set::aos_interface::part(size_t index) const {
   return parts_.pptr_[index];
}

inline particle& particle_set::aos_interface::part(size_t index) {
   return parts_.pptr_[index];
}

inline particle_set::soa_interface::soa_interface(particle_set &parts) :
      parts_(parts) {
   assert(parts_.format_ == format::soa);
}

inline fixed32 particle_set::soa_interface::pos(int dim, size_t index) const {
   return parts_.xptr_[dim][index - parts_.offset_];
}

inline float particle_set::soa_interface::vel(int dim, size_t index) const {
   return parts_.vptr_[dim][index - parts_.offset_];
}

inline rung_t particle_set::soa_interface::rung(size_t index) const {
   return parts_.rptr_[index - parts_.offset_];
}

inline fixed32& particle_set::soa_interface::pos(int dim, size_t index) {
   return parts_.xptr_[dim][index - parts_.offset_];
}

inline float& particle_set::soa_interface::vel(int dim, size_t index) {
   return parts_.vptr_[dim][index - parts_.offset_];
}

inline rung_t& particle_set::soa_interface::rung(size_t index) {
   return parts_.rptr_[index - parts_.offset_];
}

inline particle particle_set::soa_interface::part(size_t index) const {
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
