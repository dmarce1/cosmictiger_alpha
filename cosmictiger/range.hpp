#pragma once

#include <cosmictiger/defs.hpp>
#include <cosmictiger/fixed.hpp>

#include <array>

struct range {
   std::array<fixed64,NDIM> begin;
   std::array<fixed64,NDIM> end;

   inline bool contains(std::array<fixed32,NDIM> v) const {
      bool rc = true;
      for( int dim = 0; dim < NDIM; dim++) {
         if( fixed64(v[dim]) < begin[dim] ||fixed64( v[dim]) > end[dim]) {
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
