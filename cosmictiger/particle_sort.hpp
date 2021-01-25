/*
 * particle_source.hpp
 *
 *  Created on: Jan 25, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_PARTICLE_SORT_HPP_
#define COSMICTIGER_PARTICLE_SORT_HPP_

#include <cosmictiger/particle.hpp>

class particle_sort: public particle_set {
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
public:
   static std::vector<particle> get_sort_parts(const std::vector<particle> &lo_parts, int dim, fixed32 xdim);
   static size_t remote_sort(std::vector<count_t>, size_t, size_t, int, fixed32);
   static size_t sort(size_t, size_t, int, fixed32);
   static std::vector<count_t> get_count(size_t, size_t, int, fixed32);
   static size_t local_sort(size_t,size_t,int,fixed32);

};

#endif /* COSMICTIGER_PARTICLE_SORT_HPP_ */
