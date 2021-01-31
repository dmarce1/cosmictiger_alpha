#pragma once

#include <cosmictiger/defs.hpp>
#include <cosmictiger/fixed.hpp>

#include <array>

#define NCORNERS (1<<NDIM)

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
   inline std::array<std::array<fixed64,NDIM>,NCORNERS> get_corners() {
      std::array<std::array<fixed64,NDIM>,NCORNERS> v;
      for( int ci = 0; ci < NCORNERS; ci++) {
         for( int dim = 0; dim < NDIM; dim++) {
            if( (ci >> 1) & 1 ) {
               v[ci][dim] = begin[dim];
            } else {
               v[ci][dim] = end[dim];
            }
         }
      }
      return v;
   }

};
