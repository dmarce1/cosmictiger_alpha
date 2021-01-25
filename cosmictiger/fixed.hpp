/*
 * fixed.hpp
 *
 *  Created on: Jan 23, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_FIXED_HPP_
#define COSMICTIGER_FIXED_HPP_

#include <cstdint>

template<class>
class fixed;

using fixed32 = fixed<int32_t>;
using fixed64 = fixed<int64_t>;

template<class T>
class fixed {
   T i;
public:
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
