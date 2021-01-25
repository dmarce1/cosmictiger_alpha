/*
 * fixed.hpp
 *
 *  Created on: Jan 23, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_FIXED_HPP_
#define COSMICTIGER_FIXED_HPP_

#include <cstdint>

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

};

template<class T>
inline void swap(fixed<T> &first, fixed<T> &second) {
   first.i ^= second.i;
   second.i ^= first.i;
   first.i ^= second.i;
}

using fixed32 = fixed<int32_t>;

#endif /* COSMICTIGER_FIXED_HPP_ */
