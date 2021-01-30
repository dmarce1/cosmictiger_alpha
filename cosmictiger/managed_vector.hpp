/*
 * cuda_vector.hpp
 *
 *  Created on: Jan 30, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_MANAGED_VECTOR_HPP_
#define COSMICTIGER_MANAGED_VECTOR_HPP_

#include <cosmictiger/memory.hpp>

template<class T>
class managed_vector {
   T* data;
   size_t size;
   size_t capacity;

   void allocate(size_t new_size) {
      if( size != new_size) {

      }
   }

public:
   managed_vector() {
      size = 0;
      data = nullptr;
      capacity = 0;
   }
};



#endif /* COSMICTIGER_MANAGED_VECTOR_HPP_ */
