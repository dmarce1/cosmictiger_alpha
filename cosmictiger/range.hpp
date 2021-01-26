#pragma once

#include <cosmictiger/defs.hpp>
#include <cosmictiger/fixed.hpp>

struct range {
   std::array<fixed32,NDIM> begin;
   std::array<fixed32,NDIM> end;

   inline bool contains(std::array<fixed32,NDIM> v) {
      bool rc = true;
      for( int dim = 0; dim < NDIM; dim++) {
         if( v[dim] < begin[dim] || v[dim] > end[dim]) {
            rc = false;
            break;
         }
      }
      return rc;
   }
   template<class A>
   inline void serialize(A&& arc, unsigned) {
      arc & begin;
      arc & end;
   }

};
