/*
 * fixed.hpp
 *
 *  Created on: Jan 23, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_FIXED_HPP_
#define COSMICTIGER_FIXED_HPP_

#include <cstdint>

template<class >
class fixed;

using fixed32 = fixed<int32_t>;
using fixed64 = fixed<int64_t>;

#include <cassert>
#include <cstdlib>

template<class T>
class fixed {
   T i;
   static constexpr float c0 = float(size_t(1) << size_t(32));
   static constexpr float cinv = 1.f / c0;
public:
   inline fixed<T>() = default;
   inline fixed<T>(float number) {
      assert(number >= -0.5);
      assert(number < 0.5);
      i = c0 * number;
   }
   inline bool operator>(fixed other) const {
      return i > other.i;
   }
   inline bool operator<=(fixed other) const {
      return i <= other.i;
   }

   template<class A>
   void serialize(A &arc, unsigned) {
      arc & i;
   }

   template<class V>
   friend void swap(fixed<V> &first, fixed<V> &second);

   friend fixed32 rand_fixed32();

};

template<class T>
inline void swap(fixed<T> &first, fixed<T> &second) {
   first.i ^= second.i;
   second.i ^= first.i;
   first.i ^= second.i;
}

#endif /* COSMICTIGER_FIXED_HPP_ */
