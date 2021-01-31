#pragma once

#include <immintrin.h>

#include <cassert>
#include <climits>

#include <cosmictiger/fixed.hpp>

class simd_double;

class simd_int64 {
   int64_t i __attribute__ ((vector_size (sizeof(int64_t)*4)));
public:
   simd_int64() = default;
   simd_int64(const simd_int64&) = default;
   simd_int64(simd_int64&&) = default;
   simd_int64& operator=(const simd_int64&) = default;
   simd_int64& operator=(simd_int64&&) = default;

   static inline size_t size() {
      return 4;
   }

   simd_int64(simd_double r);

   inline simd_int64(int64_t j) {
      for (int k = 0; k < size(); k++) {
         i[k] = j;
      }
   }

   inline simd_int64 operator+(const simd_int64 &other) const {
      simd_int64 res;
      res.i = (i + other.i);
      return res;
   }

   inline simd_int64 operator-(const simd_int64 &other) const {
      simd_int64 res;
      res.i = (i - other.i);
      return res;
   }

   inline simd_int64 operator*(const simd_int64 &other) const {
      simd_int64 res;
      res.i = i * (other.i);
      return res;
   }

   inline simd_int64& operator+=(const simd_int64 &other) {
      *this = *this + other;
      return *this;
   }

   inline simd_int64& operator-=(const simd_int64 &other) {
      *this = *this - other;
      return *this;
   }

   inline simd_int64& operator*=(const simd_int64 &other) {
      *this = *this * other;
      return *this;
   }

   inline simd_int64 operator<<(int64_t shift) const {
      return *this * simd_int64(1LL << shift);
   }

   inline simd_int64& operator<<=(const int64_t &other) {
      *this = *this << other;
      return *this;
   }

   inline int64_t operator[](int j) const {
      assert(j >= 0);
      assert(j < size());
      return i[j];
   }

   inline int64_t& operator[](int j) {
      assert(j >= 0);
      assert(j < size());
      return reinterpret_cast<int64_t*>(&i)[j];
   }

   friend class simd_double;
};

class simd_double {
   double r __attribute__ ((vector_size (sizeof(double)*4)));
public:
   simd_double() = default;
   simd_double(const simd_double&) = default;
   simd_double(simd_double&&) = default;
   simd_double& operator=(const simd_double&) = default;
   simd_double& operator=(simd_double&&) = default;

   static inline size_t size() {
      return 4;
   }

   inline simd_double(double j) {
      for (int k = 0; k < size(); k++) {
         r[k] = j;
      }
   }

   inline double operator[](int j) const {
      assert(j >= 0);
      assert(j < size());
      return r[j];
   }

   inline double& operator[](int j) {
      assert(j >= 0);
      assert(j < size());
      return r[j];
   }

   inline simd_double operator+(const simd_double &other) const {
      simd_double res;
      res.r = _mm256_add_pd(r, other.r);
      return res;
   }

   inline simd_double operator-(const simd_double &other) const {
      simd_double res;
      res.r = _mm256_sub_pd(r, other.r);
      return res;
   }

   inline simd_double operator*(const simd_double &other) const {
      simd_double res;
      res.r = _mm256_mul_pd(r, other.r);
      return res;
   }

   inline simd_double operator/(const simd_double &other) const {
      simd_double res;
      res.r = _mm256_div_pd(r, other.r);
      return res;
   }

   inline simd_double& operator+=(const simd_double &other) {
      *this = *this + other;
      return *this;
   }

   inline simd_double& operator-=(const simd_double &other) {
      *this = *this - other;
      return *this;
   }

   inline simd_double& operator*=(const simd_double &other) {
      *this = *this * other;
      return *this;
   }

   inline simd_double& operator/=(const simd_double &other) {
      *this = *this / other;
      return *this;
   }
   inline double sum() const {
      double s = 0.0;
      for (int i = 0; i < size(); i++) {
         s += (*this)[i];
      }
      return s;
   }

   simd_double(simd_int64 i64);

   friend class simd_int64;
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

   inline simd_float(float j) {
      for (int k = 0; k < size(); k++) {
         r[k] = j;
      }
   }

   inline double operator[](int j) const {
      assert(j >= 0);
      assert(j < size());
      return r[j];
   }

   inline double& operator[](int j) {
      assert(j >= 0);
      assert(j < size());
      return reinterpret_cast<double*>(&r)[j];
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

};

class simd_fixed64 {
   simd_int64 i;
   static constexpr float c0 = float(size_t(1) << size_t(32));
   static constexpr float cinv = 1.f / c0;
   static constexpr size_t width = (sizeof(float) * CHAR_BIT);
public:

   static inline size_t size() {
      return 4;
   }

   inline fixed64 operator[](int j) const {
      assert(j >= 0);
      assert(j < size());
      return reinterpret_cast<const fixed64*>(&i)[j];
   }

   inline fixed64& operator[](int j) {
      assert(j >= 0);
      assert(j < size());
      return reinterpret_cast<fixed64*>(&i)[j];
   }

   inline simd_fixed64() = default;

   inline simd_fixed64(float number) :
         i(c0 * number) {
   }
   inline simd_double to_double() const {
      return simd_double(i) * simd_double(cinv);

   }

   inline simd_fixed64 operator+(const simd_fixed64 &other) const {
      simd_fixed64 a;
      a.i = i + other.i;
      return a;
   }

   inline simd_fixed64 operator-(const simd_fixed64 &other) const {
      simd_fixed64 a;
      a.i = i - other.i;
      return a;
   }

   inline simd_fixed64& operator+=(const simd_fixed64 &other) {
      i += other.i;
      return *this;
   }

   inline simd_fixed64& operator-=(const simd_fixed64 &other) {
      i -= other.i;
      return *this;
   }

};

#ifndef __CUDACC__
inline simd_int64::simd_int64(simd_double r) {
i = __builtin_convertvector(r.r,decltype(i));
}

inline simd_double::simd_double(simd_int64 i64) {
r = __builtin_convertvector(i64.i,decltype(r));
}


#endif
