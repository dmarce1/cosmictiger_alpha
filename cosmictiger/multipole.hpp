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
   template<class V>
   CUDA_EXPORT inline multipole_type<T> operator>>(const std::array<V,NDIM> &Y) const;
   CUDA_EXPORT inline multipole_type<T> operator +(const multipole_type<T> &vec) const;
   template<class V>
   CUDA_EXPORT inline multipole_type<T>& operator>>=(const std::array<V,NDIM> &Y);
   CUDA_EXPORT T& operator[](int i) {
      return data[i];
   }
   CUDA_EXPORT const T operator[](int i) const {
      return data[i];
   }

   template<class A>
   void serialize(A &&arc, unsigned) {
      for (int i = 0; i < MP; i++) {
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
   static constexpr size_t map3[3][3][3] = { { { 0, 1, 2 }, { 1, 3, 4 }, { 2, 4, 5 } }, { { 1, 3, 4 }, { 3, 6, 7 }, { 4,
         7, 8 } }, { { 2, 4, 5 }, { 4, 7, 8 }, { 5, 8, 9 } } };

   return data[7 + map3[i][j][k]];
}
template<class T>
CUDA_EXPORT inline T& multipole_type<T>::operator ()(int i, int j, int k) {
   static constexpr size_t map3[3][3][3] = { { { 0, 1, 2 }, { 1, 3, 4 }, { 2, 4, 5 } }, { { 1, 3, 4 }, { 3, 6, 7 }, { 4,
         7, 8 } }, { { 2, 4, 5 }, { 4, 7, 8 }, { 5, 8, 9 } } };
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

template<class T>
template<class V>
CUDA_EXPORT inline multipole_type<T> multipole_type<T>::operator>>(const std::array<V,NDIM> &dX) const {
   multipole_type you = *this;
   you >>= dX;
   return you;
}

template<class T>
CUDA_EXPORT inline multipole_type<T> multipole_type<T>::operator +(const multipole_type<T> &vec) const {
   multipole_type<T> C;
   for (int i = 0; i < MP; i++) {
      C[i] = data[i] + vec[i];
   }
   return C;
}

template<class T>
template<class V>
CUDA_EXPORT inline multipole_type<T>& multipole_type<T>::operator>>=(const std::array<V,NDIM> &Y) {
   multipole_type<T> &me = *this;
   for (int p = 0; p < 3; p++) {
      for (int q = p; q < 3; q++) {
         for (int l = q; l < 3; l++) {
            me(p, q, l) -= me() * Y[p] * Y[q] * Y[l];
            me(p, q, l) -= me(p, q) * Y[l];
            me(p, q, l) -= me(q, l) * Y[p];
            me(p, q, l) -= me(l, p) * Y[q];
         }
      }
   }
   for (int p = 0; p < 3; p++) {
      for (int q = p; q < 3; q++) {
         me(p, q) += me() * Y[p] * Y[q];
      }
   }
   return me;
}

using multipole = multipole_type<float>;

struct multi_source {
   multipole m;
   std::array<fixed32, NDIM> x;
};

#endif /* multipole_type_H_ */

