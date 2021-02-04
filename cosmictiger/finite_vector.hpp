#pragma once

#include <cosmictiger/memory.hpp>
#include <cosmictiger/cuda.hpp>
#include <cosmictiger/defs.hpp>

template<class T, size_t N>
class finite_vector {
   T *ptr;
   size_t sz;
   CUDA_EXPORT inline void destruct(size_t b, size_t e) {
      for (size_t i = b; i < e; i++) {
         ptr[i].T::~T();
      }
   }
   CUDA_EXPORT inline void construct(size_t b, size_t e) {
      for (int i = b; i < e; i++) {
         new (ptr + i) T();
      }
   }
public:
   CUDA_EXPORT inline finite_vector() {
      CUDA_MALLOC(ptr, N);
      sz = 0;
   }
   CUDA_EXPORT inline finite_vector(const finite_vector &other) = delete;
   CUDA_EXPORT inline finite_vector(finite_vector &&other) = delete;
   CUDA_EXPORT inline ~finite_vector() {
      destruct(0, sz);
      CUDA_FREE(ptr);
   }
   CUDA_EXPORT inline finite_vector& operator=(const finite_vector &other) {
      destruct(0, sz);
      sz = other.sz;
      construct(0, sz);
      for (size_t i = 0; i < sz; i++) {
         ptr[i] = other.ptr[i];
      }
      return *this;
   }
   CUDA_EXPORT inline finite_vector& operator=(finite_vector &&other) {
      sz = 0;
      swap(other);
      return *this;
   }
   CUDA_EXPORT inline size_t size() const {
      return sz;
   }
   CUDA_EXPORT inline void resize(size_t _sz) {
      assert(_sz <= N);
      construct(sz, _sz);
      destruct(_sz, sz);
      sz = _sz;
   }
   CUDA_EXPORT inline void push_back(const T &data) {
      new (ptr + sz++) T(data);
   }
   CUDA_EXPORT inline void push_back(T &&data) {
      new (ptr + sz++) T(std::move(data));
   }
   CUDA_EXPORT inline T& back() {
      BOUNDS_CHECK2(0, sz);
      return ptr[sz - 1];
   }
   CUDA_EXPORT inline const T& back() const {
      BOUNDS_CHECK2(0, sz);
      return ptr[sz - 1];
   }
   CUDA_EXPORT inline T& front() {
      BOUNDS_CHECK2(0, sz);
      return ptr[0];
   }
   CUDA_EXPORT inline const T& front() const {
      BOUNDS_CHECK2(0, sz);
      return ptr[0];
   }
   CUDA_EXPORT inline T& operator[](size_t i) {
      BOUNDS_CHECK1(i, 0, sz);
      return ptr[i];
   }
   CUDA_EXPORT inline const T& operator[](size_t i) const {
      BOUNDS_CHECK1(i, 0, sz);
      return ptr[i];
   }
   CUDA_EXPORT inline void swap(finite_vector &other) {
      const auto tmp1 = other.sz;
      const auto tmp3 = other.ptr;
      other.sz = sz;
      other.ptr = ptr;
      sz = tmp1;
      ptr = tmp3;

   }
};
