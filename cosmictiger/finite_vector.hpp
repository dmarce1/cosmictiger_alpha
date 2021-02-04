#pragma once

#include <cosmictiger/hpx.hpp>
#include <cosmictiger/memory.hpp>
#include <cosmictiger/cuda.hpp>
#include <cosmictiger/defs.hpp>

#include <stack>

template<size_t SIZE>
class finite_vector_allocator {
   static constexpr size_t page_size = ALLOCATION_PAGE_SIZE / SIZE;
   thread_local static std::vector<int8_t*> allocs;
   thread_local static std::stack<int8_t*> freelist;
public:
   void* allocate() {
      if (freelist.empty()) {
         int8_t *ptr;
         CUDA_MALLOC(ptr, page_size * SIZE);
         allocs.push_back(ptr);
         for (int i = 0; i < page_size; i++) {
            freelist.push(ptr + i * SIZE);
         }
      }
      int8_t *ptr = freelist.top();
      freelist.pop();
      return ptr;
   }
   ~finite_vector_allocator() {
      for( int i = 0; i < allocs.size(); i++) {
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


template<class T, size_t N>
thread_local finite_vector_allocator<sizeof(T) * N> finite_vector<T, N>::alloc;
