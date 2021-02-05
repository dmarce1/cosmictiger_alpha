#pragma once

#include <cosmictiger/hpx.hpp>
#include <cosmictiger/memory.hpp>
#include <cosmictiger/cuda.hpp>
#include <cosmictiger/defs.hpp>

#include <stack>

#ifdef __CUDA_ARCH__
#define BLOCKSIZE const int& blocksize = blockDim.x
#define THREADID const int& tid = threadIdx.x
#else
#define BLOCKSIZE constexpr int blocksize = 1
#define THREADID constexpr int tid = 0
#endif

template<size_t SIZE>
class finite_vector_allocator {
   static constexpr size_t page_size = 65536;
   thread_local static std::vector<int8_t*> allocs;
   thread_local static std::stack<int8_t*> freelist;
public:
   finite_vector_allocator() {
   }
   void* allocate() {
      if (freelist.empty()) {
         int8_t *ptr;
         CUDA_MALLOC(ptr, page_size * SIZE);
         // printf( "%li\n", page_size*SIZE);
         allocs.push_back(ptr);
         for (int i = 0; i < page_size - 1; i++) {
            freelist.push(ptr + i * SIZE);
         }
      }
      int8_t *ptr = freelist.top();
      freelist.pop();
      return ptr;
   }
   ~finite_vector_allocator() {
      for (int i = 0; i < allocs.size(); i++) {
         //       CUDA_FREE(allocs[i]);
      }
   }
   void deallocate(void *ptr) {
      freelist.push((int8_t*) ptr);
   }

};

template<size_t SIZE>
thread_local std::vector<int8_t*> finite_vector_allocator<SIZE>::allocs;

template<size_t SIZE>
thread_local std::stack<int8_t*> finite_vector_allocator<SIZE>::freelist;

template<class T, size_t N>
class finite_vector {
   T *ptr;
   static thread_local finite_vector_allocator<sizeof(T) * N> alloc;
   size_t sz;
   CUDA_EXPORT inline void destruct(size_t b, size_t e) {
      BLOCKSIZE;
      THREADID;
      for (size_t i = b + tid; i < e; i += blocksize) {
         ptr[i].T::~T();
      }
      CUDA_SYNC();
   }
   CUDA_EXPORT inline void construct(size_t b, size_t e) {
      BLOCKSIZE;
      THREADID;
      for (int i = b + tid; i < e; i += blocksize) {
         new (ptr + i) T();
      }
      CUDA_SYNC();
   }
public:
   CUDA_EXPORT inline finite_vector() {
#ifndef __CUDA_ARCH__
      ptr = (T*) alloc.allocate();
#else
      assert(false);
#endif
      sz = 0;
   }
   CUDA_EXPORT inline finite_vector(const finite_vector &other) = delete;
   CUDA_EXPORT inline finite_vector(finite_vector &&other) {
#ifndef __CUDA_ARCH__
      ptr = (T*) alloc.allocate();
#else
      assert(false);
#endif
      sz = 0;
      swap(other);
   }
   CUDA_EXPORT inline ~finite_vector() {
      destruct(0, sz);
#ifndef __CUDA_ARCH__
      alloc.deallocate(ptr);
#else
      assert(false);
#endif
   }
   CUDA_EXPORT inline finite_vector& operator=(const finite_vector &other) {
      BLOCKSIZE;
      THREADID;
      destruct(0, sz);
      if (tid == 0) {
         sz = other.sz;
      }
      CUDA_SYNC();
      construct(0, sz);
      for (size_t i = tid; i < sz; i += blocksize) {
         ptr[i] = other.ptr[i];
      }
      CUDA_SYNC();
      return *this;
   }
   CUDA_EXPORT inline finite_vector& operator=(finite_vector &&other) {
      THREADID;
      if (tid == 0) {
         sz = 0;
      }
      CUDA_SYNC();
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
      THREADID;
      if (tid == 0) {
         sz = _sz;
      } CUDA_SYNC();
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
      THREADID;
      if (tid == 0) {
         const auto tmp1 = other.sz;
         const auto tmp3 = other.ptr;
         other.sz = sz;
         other.ptr = ptr;
         sz = tmp1;
         ptr = tmp3;
      }
      CUDA_SYNC();
   }
};

template<class T, size_t N>
thread_local finite_vector_allocator<sizeof(T) * N> finite_vector<T, N>::alloc;
