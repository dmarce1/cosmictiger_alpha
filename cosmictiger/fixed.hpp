/*
 * fixed.hpp
 *
 *  Created on: Jan 23, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_FIXED_HPP_
#define COSMICTIGER_FIXED_HPP_

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/defs.hpp>

#include <cstdint>
#include <limits>
#include <array>

template<class >
class fixed;

using morton_t = uint64_t;

using fixed32 = fixed<uint32_t>;
using fixed64 = fixed<uint64_t>;

#include <cassert>
#include <cstdlib>

template<class T>
class fixed {
   T i;
   static constexpr float c0 = float(size_t(1) << size_t(32));
   static constexpr float cinv = 1.f / c0;
   static constexpr T width = (sizeof(float) * CHAR_BIT);
public:

   inline static fixed<T> max() {
      fixed<T> num;
      num.i = std::numeric_limits < T > ::max();
      return num;
   }
   inline static fixed<T> min() {
      fixed<T> num;
      num.i = std::numeric_limits < T > ::min();
      return num;
   }

   inline fixed<T>() = default;

   inline
#ifdef NDEBUG
   constexpr
#endif
   fixed<T>(float number) :
         i(c0 * number) {
   }
   template<class V>

   inline constexpr fixed<T>(fixed<V> other) :
         i(other.i) {
   }

   inline bool operator<(fixed other) const {
      return i < other.i;
   }

   inline bool operator>(fixed other) const {
      return i > other.i;
   }

   inline bool operator<=(fixed other) const {
      return i <= other.i;
   }

   inline bool operator>=(fixed other) const {
      return i >= other.i;
   }

   inline bool operator==(fixed other) const {
      return i == other.i;
   }

   inline bool operator!=(fixed other) const {
      return i != other.i;
   }

   inline float to_float() const {
      return i * cinv;

   }

   inline int to_int() const {
      return i >> width;
   }

   inline double to_double() const {
      return i * cinv;

   }

   inline fixed<T> operator*(const fixed<T> &other) const {
      int64_t a;
      const int64_t b = i;
      const int64_t c = other.i;
      a = (b * c) >> width;
      fixed<T> res;
      res.i = (T) a;
      return res;
   }

   inline fixed<T> operator*=(const fixed<T> &other) {
      int64_t a;
      const int64_t b = i;
      const int64_t c = other.i;
      a = (b * c) >> width;
      i = (T) a;
      return *this;
   }

   inline fixed<T> operator*=(int other) {
      int64_t a;
      const int64_t b = i;
      const int64_t c = other;
      a = b * c;
      i = (T) a;
      return *this;
   }

   inline fixed<T> operator/(const fixed<T> &other) const {
      int64_t a;
      const int64_t b = i;
      const int64_t c = other.i;
      a = b / (c >> width);
      fixed<T> res;
      res.i = (T) a;
      return res;
   }

   inline fixed<T> operator/=(const fixed<T> &other) {
      int64_t a;
      const int64_t b = i;
      const int64_t c = other.i;
      a = b / (c >> width);
      i = (T) a;
      return *this;
   }

   inline fixed<T> operator+(const fixed<T> &other) const {
      fixed<T> a;
      a.i = i + other.i;
      return a;
   }

   inline fixed<T> operator-(const fixed<T> &other) const {
      fixed<T> a;
      a.i = i - other.i;
      return a;
   }

   inline fixed<T>& operator+=(const fixed<T> &other) {
      i += other.i;
      return *this;
   }

   inline fixed<T>& operator-=(const fixed<T> &other) {
      i -= other.i;
      return *this;
   }

   friend morton_t morton_key(std::array<fixed32, NDIM> num, int64_t);

   template<class A>
   void serialize(A &arc, unsigned) {
      arc & i;
   }

   template<class >
   friend class fixed;

   template<class V>
   friend void swap(fixed<V> &first, fixed<V> &second);

   friend fixed32 rand_fixed32();

};

inline morton_t morton_key(std::array<fixed32, NDIM> I, int64_t depth) {
   assert(depth % NDIM == 0);
   morton_t key = 0LL;
   for (size_t dim = 0; dim < NDIM; dim++) {
      I[dim].i >>= (sizeof(morton_t) * CHAR_BIT - depth);
   }
   for (size_t k = 0; k < depth / NDIM; k++) {
      for (size_t dim = 0; dim < NDIM; dim++) {
         key ^= size_t((bool) (I[dim].i & (0x0000000000000001LL << k))) << size_t(k * NDIM + dim);
      }
   }
   return key;
}

template<class T>
inline void swap(fixed<T> &first, fixed<T> &second) {
   std::swap(first, second);
}

#endif /* COSMICTIGER_FIXED_HPP_ */
