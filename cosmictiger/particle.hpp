/*
 * particle.hpp
 *
 *  Created on: Jan 23, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_PARTICLE_HPP_
#define COSMICTIGER_PARTICLE_HPP_

#include <cosmictiger/fixed.hpp>
#include <cosmictiger/vect3.hpp>
#include <cosmictiger/hpx.hpp>

#include <array>
#include <atomic>
#include <vector>

struct particle {
   std::array<fixed32, NDIM> x;
   std::array<float, NDIM> v;
   int8_t rung;
   template<class A>
   void serialize(A &&arc, unsigned) {
      arc & x;
      arc & v;
      arc & rung;
   }
};

struct particle_set {
protected:
   struct members_t {
      std::array<fixed32*, NDIM> x;
      std::array<float*, NDIM> v;
      int8_t *rung;
      size_t size;
      int64_t offset;
      bool virtual_;
   };
   members_t *mems;
   static members_t parts;
   static int index_to_rank(size_t);
   static std::pair<size_t,size_t> rank_to_range(int);
   static std::pair<hpx::id_type, hpx::id_type> rel_children(size_t, size_t);
public:
   inline particle_set() {
      mems = nullptr;
   }
   static void random_particle_set();
   static void create();
   static void destroy();
};



#endif /* COSMICTIGER_PARTICLE_HPP_ */
