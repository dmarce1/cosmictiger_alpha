#pragma once

#include <immintrin.h>

class uint64_simd {
   __m256i i;
public:
   uint64_simd() = default;
   uint64_simd(const uint64&) = default;
   uint64_simd(uint64&&) = default;
   uint64_simd& operator=(const uint64&) = default;
   uint64_simd& operator=(uint64&&) = default;

   inline uint64_simd(uint64_t j) {
      i = _mm256_set_epi64x(j, j, j, j);
   }

   inline uint64_simd operator+(const uint64_simd &other) const {
      uint64_simd res;
      res.i = _mm256_add_epi64(i, other.i);
      return res;
   }

   inline uint64_simd operator-(const uint64_simd &other) const {
      uint64_simd res;
      res.i = _mm256_sub_epi64(i, other.i);
      return res;
   }

   inline uint64_simd operator*(const uint64_simd &other) const {
      uint64_simd res;
      res.i = _mm256_mullo_epi64(i, other.i);
      return res;
   }

   inline uint64_simd operator/(const uint64_simd &other) const {
      uint64_simd res;
      res.i = _mm256_div_epi64(i, other.i);
      return res;
   }

   inline uint64_simd& operator+=(const uint64_simd &other) {
      *this = *this + other;
      return *this;
   }

   inline uint64_simd& operator-=(const uint64_simd &other) {
      *this = *this - other;
      return *this;
   }

   inline uint64_simd& operator*=(const uint64_simd &other) {
      *this = *this * other;
      return *this;
   }

   inline uint64_simd& operator/=(const uint64_simd &other) {
      *this = *this / other;
      return *this;
   }

   inline uint64_simd operator>>(uint64_t shift) {
      return *this / uint64_simd(1LL << shift);
   }

   inline uint64_simd operator<<(uint64_t shift) {
      return *this * uint64_simd(1LL << shift);
   }

   inline uint64_simd& operator>>=(const uint64_simd &other) {
      *this = *this >> other;
      return *this;
   }

   inline uint64_simd& operator<<=(const uint64_simd &other) {
      *this = *this << other;
      return *this;
   }

};
