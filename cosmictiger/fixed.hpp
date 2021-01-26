/*
 * fixed.hpp
 *
 *  Created on: Jan 23, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_FIXED_HPP_
#define COSMICTIGER_FIXED_HPP_

#include <cstdint>
#include <limits>
#include <cosmictiger/cuda.hpp>

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
   CUDA_EXPORT
   inline static fixed<T> max() {
      fixed<T> num;
      num.i = std::numeric_limits<T>::max();
      return num;
   }
   inline static fixed<T> min() {
      fixed<T> num;
      num.i = std::numeric_limits<T>::min();
      return num;
   }
   CUDA_EXPORT
   inline fixed<T>() = default;
   CUDA_EXPORT
   inline
#ifdef NDEBUG
   constexpr
#endif
   fixed<T>(float number) :
         i(c0 * number) {
      assert(number >= -0.5);
      assert(number < 0.5);
   }
   template<class V>
   CUDA_EXPORT
   inline constexpr fixed<T>(fixed<V> other) :
         i(other.i) {
   }
   CUDA_EXPORT
   inline bool operator<(fixed other) const {
      return i < other.i;
   }
   CUDA_EXPORT
   inline bool operator>(fixed other) const {
      return i > other.i;
   }
   CUDA_EXPORT
   inline bool operator<=(fixed other) const {
      return i <= other.i;
   }
   CUDA_EXPORT
   inline bool operator>=(fixed other) const {
      return i >= other.i;
   }
   CUDA_EXPORT
   inline bool operator==(fixed other) const {
      return i == other.i;
   }
   CUDA_EXPORT
   inline bool operator!=(fixed other) const {
      return i != other.i;
   }
   CUDA_EXPORT
   inline float to_float() const {
      return i * cinv;

   }
   CUDA_EXPORT
   inline double to_double() const {
      return i * cinv;

   }
   CUDA_EXPORT
   inline fixed<T> operator*(const fixed<T> &other) const {
      int64_t a;
      const int64_t b = i;
      const int64_t c = other.i;
      a = (b * c) >> (sizeof(float) * CHAR_BIT);
      fixed<T> res;
      res.i = (T) a;
      return res;
   }
   CUDA_EXPORT
   inline fixed<T> operator*=(const fixed<T> &other) const {
      int64_t a;
      const int64_t b = i;
      const int64_t c = other.i;
      a = (b * c) >> (sizeof(float) * CHAR_BIT);
      i = (T) a;
      return *this;
   }
   CUDA_EXPORT
   inline fixed<T> operator/(const fixed<T> &other) const {
      int64_t a;
      const int64_t b = i;
      const int64_t c = other.i;
      a = b / (c >> (sizeof(float) * CHAR_BIT));
      fixed<T> res;
      res.i = (T) a;
      return res;
   }
   CUDA_EXPORT
   inline fixed<T> operator/=(const fixed<T> &other) const {
      int64_t a;
      const int64_t b = i;
      const int64_t c = other.i;
      a = b / (c >> (sizeof(float) * CHAR_BIT));
      i = (T) a;
      return *this;
   }

   CUDA_EXPORT
   inline fixed<T> operator+(const fixed<T> &other) const {
      fixed<T> a;
      a.i = i + other.i;
      return a;
   }
   CUDA_EXPORT
   inline fixed<T> operator-(const fixed<T> &other) const {
      fixed<T> a;
      a.i = i - other.i;
      return a;
   }
   CUDA_EXPORT
   inline fixed<T>& operator+=(const fixed<T> &other) {
      i += other.i;
      return *this;
   }
   CUDA_EXPORT
   inline fixed<T>& operator-=(const fixed<T> &other) {
      i -= other.i;
      return *this;
   }

   template<class A>
   void serialize(A &arc, unsigned) {
      arc & i;
   }

   template<class>
   friend class fixed;

   template<class V>
   friend void swap(fixed<V> &first, fixed<V> &second);

   friend fixed32 rand_fixed32();

};

template<class T>
CUDA_EXPORT
inline void swap(fixed<T> &first, fixed<T> &second) {
   std::swap(first,second);
}

#endif /* COSMICTIGER_FIXED_HPP_ */
