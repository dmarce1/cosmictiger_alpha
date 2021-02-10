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
   friend class simd_fixed32;

   CUDA_EXPORT
   inline static fixed<T> max() {
      fixed<T> num;
      num.i = std::numeric_limits < T > ::max();
      return num;
   }
   CUDA_EXPORT
   inline static fixed<T> min() {
      fixed<T> num;
      num.i = 1;
      return num;
   }

   CUDA_EXPORT
   constexpr inline fixed<T>() : i(0) {}

   CUDA_EXPORT
   inline
#ifdef NDEBUG
   constexpr
#endif
   CUDA_EXPORT
   fixed<T>& operator=(double number) {
      i = (c0 * number);
      return *this;
   }

   fixed<T>(float number) :
         i(c0 * number) {
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
      return float(i) * cinv;

   }

   CUDA_EXPORT
   inline int to_int() const {
      return i >> width;
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
      a = (b * c) >> width;
      fixed<T> res;
      res.i = (T) a;
      return res;
   }

   CUDA_EXPORT
   inline fixed<T> operator*=(const fixed<T> &other) {
      int64_t a;
      const int64_t b = i;
      const int64_t c = other.i;
      a = (b * c) >> width;
      i = (T) a;
      return *this;
   }

   CUDA_EXPORT
   inline fixed<T> operator*=(int other) {
      int64_t a;
      const int64_t b = i;
      const int64_t c = other;
      a = b * c;
      i = (T) a;
      return *this;
   }

   CUDA_EXPORT
   inline fixed<T> operator/(const fixed<T> &other) const {
      int64_t a;
      const int64_t b = i;
      const int64_t c = other.i;
      a = b / (c >> width);
      fixed<T> res;
      res.i = (T) a;
      return res;
   }

   CUDA_EXPORT
   inline fixed<T> operator/=(const fixed<T> &other) {
      int64_t a;
      const int64_t b = i;
      const int64_t c = other.i;
      a = b / (c >> width);
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

   CUDA_EXPORT
   CUDA_EXPORT
   inline T get_integer() const {
      return i;
   }

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

//Morton encoding adopted from https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/
CUDA_EXPORT
CUDA_EXPORT
inline uint64_t split3(uint64_t a) {
   uint64_t x = a & 0x1fffff;
   x = (x | x << 32) & 0x1f00000000ffff;
   x = (x | x << 16) & 0x1f0000ff0000ff;
   x = (x | x << 8) & 0x100f00f00f00f00f;
   x = (x | x << 4) & 0x10c30c30c30c30c3;
   x = (x | x << 2) & 0x1249249249249249;
   return x;
}

CUDA_EXPORT
CUDA_EXPORT
inline uint64_t morton_magicbits(uint64_t x, uint64_t y, uint64_t z) {
   uint64_t answer = 0;
   answer |= split3(x) | split3(y) << 1 | split3(z) << 2;
   return answer;
}

template<class T>
CUDA_EXPORT
inline morton_t morton_key(T x, T y, T z, int64_t depth) {
   const int shift = sizeof(float) * CHAR_BIT - (depth + NDIM) / NDIM;
   morton_t key = morton_magicbits(z.get_integer() >> shift, y.get_integer() >> shift, x.get_integer() >> shift);
   key >>= NDIM - (depth % NDIM);
   return key;
}

template<class T>
inline morton_t morton_key(std::array<T, NDIM> I, int64_t depth) {
   return morton_key(I[0], I[1], I[2], depth);
}

template<class T>
inline void swap(fixed<T> &first, fixed<T> &second) {
   std::swap(first, second);
}

#endif /* COSMICTIGER_FIXED_HPP_ */
