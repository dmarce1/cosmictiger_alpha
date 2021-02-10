#pragma once

#include <immintrin.h>

#include <cassert>
#include <climits>

#include <cosmictiger/fixed.hpp>

class simd_double;

class simd_int32 {
   int32_t i __attribute__ ((vector_size (sizeof(int32_t)*8)));
public:
   simd_int32() = default;
   simd_int32(const simd_int32&) = default;
   simd_int32(simd_int32&&) = default;
   simd_int32& operator=(const simd_int32&) = default;
   simd_int32& operator=(simd_int32&&) = default;

   static inline size_t size() {
      return 8;
   }

   simd_int32(simd_double r);

   inline simd_int32(int32_t j) {
      for (int k = 0; k < size(); k++) {
         i[k] = j;
      }
   }

   inline simd_int32 operator+(const simd_int32 &other) const {
      simd_int32 res;
      res.i = (i + other.i);
      return res;
   }

   inline simd_int32 operator-(const simd_int32 &other) const {
      simd_int32 res;
      res.i = (i - other.i);
      return res;
   }

   inline simd_int32 operator*(const simd_int32 &other) const {
      simd_int32 res;
      res.i = i * (other.i);
      return res;
   }

   inline simd_int32& operator+=(const simd_int32 &other) {
      *this = *this + other;
      return *this;
   }

   inline simd_int32& operator-=(const simd_int32 &other) {
      *this = *this - other;
      return *this;
   }

   inline simd_int32& operator*=(const simd_int32 &other) {
      *this = *this * other;
      return *this;
   }

   inline simd_int32 operator<<(int32_t shift) const {
      return *this * simd_int32(1LL << shift);
   }

   inline simd_int32& operator<<=(const int32_t &other) {
      *this = *this << other;
      return *this;
   }

   inline int32_t operator[](int j) const {
      assert(j >= 0);
      assert(j < size());
      return i[j];
   }

   inline int32_t& operator[](int j) {
      assert(j >= 0);
      assert(j < size());
      return reinterpret_cast<int32_t*>(&i)[j];
   }
   friend class simd_float;
   friend class simd_double;
};

class simd_float {
   float r __attribute__ ((vector_size (sizeof(float)*8)));
public:
   simd_float() = default;
   simd_float(const simd_float&) = default;
   simd_float(simd_float&&) = default;
   simd_float& operator=(const simd_float&) = default;
   simd_float& operator=(simd_float&&) = default;

   static inline size_t size() {
      return 8;
   }

   inline simd_float(simd_int32 i) {
      for (int k = 0; k < size(); k++) {
         r[k] = i[k];
      }
   }

   inline simd_float(float j) {
      for (int k = 0; k < size(); k++) {
         r[k] = j;
      }
   }

   inline float operator[](int j) const {
      assert(j >= 0);
      assert(j < size());
      return r[j];
   }

   inline float& operator[](int j) {
      assert(j >= 0);
      assert(j < size());
      return reinterpret_cast<float*>(&r)[j];
   }

   inline simd_float operator+(const simd_float &other) const {
      simd_float res;
      res.r = _mm256_add_ps(r, other.r);
      return res;
   }

   inline simd_float operator-(const simd_float &other) const {
      simd_float res;
      res.r = _mm256_sub_ps(r, other.r);
      return res;
   }

   inline simd_float operator*(const simd_float &other) const {
      simd_float res;
      res.r = _mm256_mul_ps(r, other.r);
      return res;
   }

   inline simd_float operator/(const simd_float &other) const {
      simd_float res;
      res.r = _mm256_div_ps(r, other.r);
      return res;
   }

   inline simd_float& operator+=(const simd_float &other) {
      *this = *this + other;
      return *this;
   }

   inline simd_float& operator-=(const simd_float &other) {
      *this = *this - other;
      return *this;
   }

   inline simd_float& operator*=(const simd_float &other) {
      *this = *this * other;
      return *this;
   }

   inline simd_float& operator/=(const simd_float &other) {
      *this = *this / other;
      return *this;
   }

   inline float sum() const {
      float s = 0.0;
      for (int i = 0; i < size(); i++) {
         s += (*this)[i];
      }
      return s;
   }
   simd_float operator-() const {
      return simd_float(0) - *this;
   }

   simd_float operator>(simd_float other) const {
      auto i = r > other.r;
      simd_int32 res;
      res.i = i;
      return res;
   }

   friend inline simd_float operator*(float a, const simd_float &other);
   friend inline simd_float sqrt(simd_float);
   friend inline simd_float fmax(simd_float, simd_float);
};

inline simd_float fmax(simd_float a, simd_float b) {
   simd_float res;
   res.r = _mm256_max_ps(a.r, b.r);
   return res;
}

inline simd_float sqrt(simd_float r) {
   simd_float res;
   res.r = _mm256_sqrt_ps(r.r);
   return res;
}

inline simd_float operator*(float a, const simd_float &other) {
   simd_float res;
   res.r = _mm256_mul_ps(other.r, simd_float(a).r);
   return res;
}

class simd_fixed32 {
   simd_int32 i;
   static constexpr float c0 = float(size_t(1) << size_t(32));
   static constexpr float cinv = 1.f / c0;
   static constexpr size_t width = (sizeof(float) * CHAR_BIT);
public:

   inline simd_fixed32(fixed32 other) {
       for( int j = 0; j < size(); j++) {
         i[j] = other.i;
      }
   }
   static inline size_t size() {
      return 8;
   }

   inline fixed32 operator[](int j) const {
      assert(j >= 0);
      assert(j < size());
      return reinterpret_cast<const fixed32*>(&i)[j];
   }

   inline fixed32& operator[](int j) {
      assert(j >= 0);
      assert(j < size());
      return reinterpret_cast<fixed32*>(&i)[j];
   }

   inline simd_fixed32() = default;

   inline simd_fixed32(float number) :
         i(c0 * number) {
   }
   inline simd_float to_float() const {
      return simd_float(i) * simd_float(cinv);

   }

   inline simd_fixed32 operator+(const simd_fixed32 &other) const {
      simd_fixed32 a;
      a.i = i + other.i;
      return a;
   }

   inline simd_fixed32 operator-(const simd_fixed32 &other) const {
      simd_fixed32 a;
      a.i = i - other.i;
      return a;
   }

   inline simd_fixed32& operator+=(const simd_fixed32 &other) {
      i += other.i;
      return *this;
   }

   inline simd_fixed32& operator-=(const simd_fixed32 &other) {
      i -= other.i;
      return *this;
   }

};

