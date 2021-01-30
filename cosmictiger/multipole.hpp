/*
 * multipole_type.hpp
 *
 *  Created on: Jan 30, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_multipole_type_HPP_
#define COSMICTIGER_multipole_type_HPP_

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/fixed.hpp>
#include <array>

constexpr int MP = 17;

template<class T>
class multipole_type {
private:
   T data[MP];
public:
   CUDA_EXPORT multipole_type();
   CUDA_EXPORT T operator ()() const;
   CUDA_EXPORT T& operator ()();
   CUDA_EXPORT T operator ()(int i, int j) const;
   CUDA_EXPORT T& operator ()(int i, int j);
   CUDA_EXPORT T operator ()(int i, int j, int k) const;
   CUDA_EXPORT T& operator ()(int i, int j, int k);
   CUDA_EXPORT multipole_type<T>& operator =(const multipole_type<T> &other);
   CUDA_EXPORT multipole_type<T>& operator =(T other);
   CUDA_EXPORT T& operator[](int i) {
      return data[i];
   }
   CUDA_EXPORT const T operator[](int i) const {
      return data[i];
   }

   template<class A>
   void serialize( A&& arc, unsigned ) {
      for( int i = 0; i < MP; i++) {
         arc & data[i];
      }
   }

};

template<class T>
CUDA_EXPORT inline multipole_type<T>::multipole_type() {
}

template<class T>
CUDA_EXPORT inline T multipole_type<T>::operator ()() const {
   return data[0];
}

template<class T>
CUDA_EXPORT inline T& multipole_type<T>::operator ()() {
   return data[0];
}

template<class T>
CUDA_EXPORT inline T multipole_type<T>::operator ()(int i, int j) const {
   static constexpr size_t map2[3][3] = { { 0, 1, 2 }, { 1, 3, 4 }, { 2, 4, 5 } };
   return data[1 + map2[i][j]];
}

template<class T>
CUDA_EXPORT inline T& multipole_type<T>::operator ()(int i, int j) {
   static constexpr size_t map2[3][3] = { { 0, 1, 2 }, { 1, 3, 4 }, { 2, 4, 5 } };
   return data[1 + map2[i][j]];
}

template<class T>
CUDA_EXPORT inline T multipole_type<T>::operator ()(int i, int j, int k) const {
   static constexpr size_t map3[3][3][3] = { { { 0, 1, 2 }, { 1, 3, 4 }, { 2, 4, 5 } }, { { 1, 3, 4 }, { 3, 6, 7 }, { 4, 7, 8 } }, { { 2, 4, 5 }, { 4, 7, 8 },
         { 5, 8, 9 } } };

   return data[7 + map3[i][j][k]];
}
template<class T>
CUDA_EXPORT inline T& multipole_type<T>::operator ()(int i, int j, int k) {
   static constexpr size_t map3[3][3][3] = { { { 0, 1, 2 }, { 1, 3, 4 }, { 2, 4, 5 } }, { { 1, 3, 4 }, { 3, 6, 7 }, { 4, 7, 8 } }, { { 2, 4, 5 }, { 4, 7, 8 },
         { 5, 8, 9 } } };
   return data[7 + map3[i][j][k]];
}


template<class T>
CUDA_EXPORT inline multipole_type<T>& multipole_type<T>::operator =(const multipole_type<T> &other) {
   memcpy(&data[0], &other.data[0], MP * sizeof(float));
   return *this;
}

template<class T>
CUDA_EXPORT inline multipole_type<T>& multipole_type<T>::operator =(T other) {
   for (int i = 0; i < MP; i++) {
      data[i] = other;
   }
   return *this;
}

using multipole = multipole_type<fixed32>;

struct multi_source {
   multipole m;
   std::array<fixed32,NDIM> x;
};


#endif /* multipole_type_H_ */

