#pragma once

#include <immintrin.h>

class int64_simd {
   __m256i i;
public:
   int64_simd() = default;
   int64_simd(const int64&) = default;
   int64_simd(int64&&) = default;
   int64_simd& operator=(const int64&) = default;
   int64_simd& operator=(int64&&) = default;

   inline int64_simd(int64_t j) {
      i = _mm256_set_epi64x(j, j, j, j);
   }

   inline int64_simd operator+(const int64_simd &other) const {
      int64_simd res;
      res.i = _mm256_add_epi64(i, other.i);
      return res;
   }

   inline int64_simd operator-(const int64_simd &other) const {
      int64_simd res;
      res.i = _mm256_sub_epi64(i, other.i);
      return res;
   }

   inline int64_simd operator*(const int64_simd &other) const {
      int64_simd res;
      res.i = _mm256_mullo_epi64(i, other.i);
      return res;
   }

   inline int64_simd operator/(const int64_simd &other) const {
      int64_simd res;
      res.i = _mm256_div_epi64(i, other.i);
      return res;
   }

   inline int64_simd& operator+=(const int64_simd &other) {
      *this = *this + other;
      return *this;
   }

   inline int64_simd& operator-=(const int64_simd &other) {
      *this = *this - other;
      return *this;
   }

   inline int64_simd& operator*=(const int64_simd &other) {
      *this = *this * other;
      return *this;
   }

   inline int64_simd& operator/=(const int64_simd &other) {
      *this = *this / other;
      return *this;
   }

   inline int64_simd operator>>(int64_t shift) const {
      return *this / int64_simd(1LL << shift);
   }

   inline int64_simd operator<<(int64_t shift) const {
      return *this * int64_simd(1LL << shift);
   }

   inline int64_simd& operator>>=(const int64_simd &other) {
      *this = *this >> other;
      return *this;
   }

   inline int64_simd& operator<<=(const int64_simd &other) {
      *this = *this << other;
      return *this;
   }

   inline uint64_simd operator^( const int64_simd& other) const {
      uint64_t res;
      res.i = _mm256_xor_epi64(i,other.i);
   }

   inline uint64_simd operator&( const int64_simd& other) const {
      uint64_t res;
      res.i = _mm256_and_epi64(i,other.i);
   }

   inline uint64_simd operator|( const int64_simd& other) const {
      uint64_t res;
      res.i = _mm256_or_epi64(i,other.i);
   }

   inline int64_simd& operator&=(const int64_simd &other) {
      *this = *this & other;
      return *this;
   }
   inline int64_simd& operator^=(const int64_simd &other) {
      *this = *this ^ other;
      return *this;
   }
   inline int64_simd& operator|=(const int64_simd &other) {
      *this = *this | other;
      return *this;
   }


};
