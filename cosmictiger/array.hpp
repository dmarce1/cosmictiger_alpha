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

template<class T, size_t N>
class array {
   T ptr[N];
public:
   CUDA_EXPORT const T& operator[](size_t i) const {
      BOUNDS_CHECK1(i,0,N);
      return ptr[i];
   }
   CUDA_EXPORT T& operator[](size_t i) {
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
