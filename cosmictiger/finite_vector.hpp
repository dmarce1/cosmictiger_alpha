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
   static constexpr size_t page_size = (65536-1 / SIZE+32);
   thread_local static std::stack<int8_t*> freelist;
   static std::stack<int8_t*> globallist;
   static std::atomic<int> lock;
public:
   finite_vector_allocator() {
   }
   void* allocate() {
      if (freelist.empty()) {
         while (lock++ != 0) {
            lock--;
         }
         if (globallist.size() < page_size) {
            lock--;
            int8_t *ptr;
            CUDA_MALLOC(ptr, page_size * SIZE);
            for (int i = 0; i < page_size; i++) {
               freelist.push(ptr + i * SIZE);
            }
         } else {
            for (int i = 0; i < page_size; i++) {
               freelist.push(globallist.top());
               globallist.pop();
            }
            lock--;
         }
      }
      int8_t *ptr = freelist.top();
      freelist.pop();
      return ptr;
   }
   ~finite_vector_allocator() {
   }
   void deallocate(void *ptr) {
      freelist.push((int8_t*) ptr);
      if (freelist.size() >= 2 * page_size) {
         while (lock++ != 0) {
            lock--;
         }
         for (int i = 0; i < page_size; i++) {
            globallist.push(freelist.top());
            freelist.pop();
         }
         lock--;
      }
   }
};

template<size_t SIZE>
std::atomic<int> finite_vector_allocator<SIZE>::lock(0);

template<size_t SIZE>
thread_local std::stack<int8_t*> finite_vector_allocator<SIZE>::freelist;

template<size_t SIZE>
std::stack<int8_t*> finite_vector_allocator<SIZE>::globallist;

template<class T, size_t N>
class finite_vector {
   T *ptr;
   size_t sz;
   static finite_vector_allocator<sizeof(T) * N> alloc;
   CUDA_EXPORT inline void destruct(size_t b, size_t e) {
      BLOCKSIZE;
      THREADID;
      for (size_t i = b + tid; i < e; i += blocksize) {
         ptr[i].T::~T();
      }CUDA_SYNC();
   }
   CUDA_EXPORT inline void construct(size_t b, size_t e) {
      BLOCKSIZE;
      THREADID;
      for (int i = b + tid; i < e; i += blocksize) {
         new (ptr + i) T();
      }CUDA_SYNC();
   }
public:
   CUDA_EXPORT inline finite_vector() {
#ifndef __CUDA_ARCH__
      THREADID;
      if (tid == 0) {
         ptr = (T*) alloc.allocate();
      }
      construct(0, N);
#else
      assert(false);
#endif
      sz = 0;
   }
   CUDA_EXPORT inline finite_vector(const finite_vector &other) = delete;
   CUDA_EXPORT inline finite_vector(finite_vector &&other) {
#ifndef __CUDA_ARCH__
      THREADID;
      if (tid == 0) {
         ptr = (T*) alloc.allocate();
      }
      construct(0, N);
#else
      assert(false);
#endif
      sz = 0;
      swap(other);
   }
   CUDA_EXPORT inline ~finite_vector() {
#ifndef __CUDA_ARCH__
      destruct(0, N);
      THREADID;
      if (tid == 0) {
         alloc.deallocate(ptr);
      }
#else
      assert(false);
#endif
   }
   CUDA_EXPORT inline finite_vector& operator=(const finite_vector &other) {
      BLOCKSIZE;
      THREADID;
      CUDA_SYNC();
      if (tid == 0) {
         sz = other.sz;
      }CUDA_SYNC();CUDA_SYNC();
      for (size_t i = tid; i < sz; i += blocksize) {
         ptr[i] = other.ptr[i];
      }CUDA_SYNC();
      return *this;
   }
   CUDA_EXPORT inline finite_vector& operator=(finite_vector &&other) {
      THREADID;
      if (tid == 0) {
         sz = 0;
      }CUDA_SYNC();
      swap(other);
      return *this;
   }
   CUDA_EXPORT inline size_t size() const {
      return sz;
   }
   CUDA_EXPORT inline void resize(size_t _sz) {
      assert(_sz <= N);
      THREADID;
      if (tid == 0) {
         sz = _sz;
      }CUDA_SYNC();
   }
   CUDA_EXPORT inline void pop_back() {
      THREADID;
      if (tid == 0) {
         sz--;
      }
   }
   CUDA_EXPORT inline T& top() {
      return ptr[sz - 1];
   }

   CUDA_EXPORT inline const T& top() const {
      return ptr[sz - 1];
   }
   CUDA_EXPORT inline void push_back(const T &data) {
      THREADID;
      if (tid == 0) {
         ptr[sz++] = data;
      }
   }
   CUDA_EXPORT inline void push_back(T &&data) {
      THREADID;
      if (tid == 0) {
         ptr[sz++] = std::move(data);
      }
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
      }CUDA_SYNC();
   }
};

template<class T, size_t N>
finite_vector_allocator<sizeof(T) * N> finite_vector<T, N>::alloc;
