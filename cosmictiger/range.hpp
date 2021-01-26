#pragma once

#include <cosmictiger/defs.hpp>
#include <cosmictiger/fixed.hpp>

struct range {
   std::array<fixed32,NDIM> begin;
   std::array<fixed32,NDIM> end;
   template<class A>
   void serialize(A&& arc, unsigned) {
      arc & begin;
      arc & end;
   }

};
