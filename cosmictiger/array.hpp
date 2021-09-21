/*
 * array.hpp
 *
 *  Created on: Feb 3, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_ARRAY_HPP_
#define COSMICTIGER_ARRAY_HPP_


#include <cassert>
#include <cosmictiger/defs.hpp>

template<class T, int N>
class array {
   T ptr[N];
public:
   const T& operator[](int i) const {
      BOUNDS_CHECK1(i,0,N);
      return ptr[i];
   }
   T& operator[](int i) {
      BOUNDS_CHECK1(i,0,N);
      return ptr[i];
   }

    T* data() {
      return ptr;
   }

   const T* data() const {
      return ptr;
   }

    T* begin() {
      return ptr;
   }

   T* end() {
      return ptr + N;
   }

   template<class A>
   void serialize(A&& arc, unsigned) {
   	for( int i = 0; i < N; i++) {
   		arc & (*this)[i];
   	}
   }

};

#endif /* COSMICTIGER_ARRAY_HPP_ */
