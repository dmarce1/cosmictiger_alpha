/*
 * particle.hpp
 *
 *  Created on: Jan 23, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_PARTICLE_HPP_
#define COSMICTIGER_PARTICLE_HPP_

#include <cosmictiger/defs.hpp>
#include <cosmictiger/fixed.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/range.hpp>

struct range;

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

class particle_set {
   std::array<fixed32*, NDIM> x;
   std::array<float*, NDIM> v;
   int8_t *rung;
   size_t size;
   int64_t offset;
   bool virtual_;
   struct count_t {
      size_t lo;
      size_t hi;
      int rank;
      bool operator<(const count_t &other) const {
         return rank < other.rank;
      }
      template<class A>
      void serialize(A &&arc, unsigned) {
         arc & lo;
         arc & hi;
         arc & rank;
      }
   };
   static std::atomic<size_t> hi_index;
   static particle_set parts;
   static std::pair<size_t, size_t> rank_to_range(int);
   static std::pair<hpx::id_type, hpx::id_type> rel_children(size_t, size_t);
public:
   inline particle_set() = default;
   fixed32 get_x(size_t index, int dim) const;
   std::array<fixed32,NDIM> get_x(size_t index) const;
   particle get_part(size_t index) const;
   void set_part(particle p, size_t index);
   static particle_set local_particle_set();
   static int index_to_rank(size_t);
   static void generate_random_particle_set();
   static void create();
   static void destroy();
   static std::vector<particle> get_sort_parts(const std::vector<particle> &lo_parts, int dim, fixed32 xdim);
//   static size_t cuda_sort(size_t, size_t, int, fixed32);
   static size_t remote_sort(std::vector<count_t>, size_t, size_t, int, fixed32);
   static size_t sort(size_t, size_t, int, fixed32);
   static std::vector<size_t> radix_sort(size_t, size_t, range box, int dimstart, int depth);
     static std::vector<count_t> get_count(size_t, size_t, int, fixed32);
   static size_t local_sort(size_t, size_t, int, fixed32);

};

#endif /* COSMICTIGER_PARTICLE_HPP_ */
