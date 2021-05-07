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
#include <cosmictiger/cuda.hpp>

template<class T, int N>
class array {
   T ptr[N];
public:

   template<class A>
   void serialize(A&& arc, unsigned) {
   	for( int i = 0; i < N; i++) {
   		arc & ptr[i];
   	}
   }

   CUDA_EXPORT const T& operator[](int i) const {
      BOUNDS_CHECK1(i,0,N);
      return ptr[i];
   }
   CUDA_EXPORT T& operator[](int i) {
      BOUNDS_CHECK1(i,0,N);
      return ptr[i];
   }

   CUDA_EXPORT  T* data() {
      return ptr;
   }

   CUDA_EXPORT const T* data() const {
      return ptr;
   }

   CUDA_EXPORT  T* begin() {
      return ptr;
   }

   CUDA_EXPORT T* end() {
      return ptr + N;
   }

};

#endif /* COSMICTIGER_ARRAY_HPP_ */
