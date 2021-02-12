#pragma once

#include <cosmictiger/hpx.hpp>
#include <cosmictiger/memory.hpp>
#include <cosmictiger/cuda.hpp>
#include <cosmictiger/defs.hpp>

#include <cassert>
#include <stack>
#include <atomic>

#ifdef __CUDA_ARCH__
#define BLOCKSIZE const int& blocksize = blockDim.x
#define THREADID const int& tid = threadIdx.x
#else
#define BLOCKSIZE constexpr int blocksize = 1
#define THREADID constexpr int tid = 0
#endif


template<size_t SIZE>
class finite_vector_allocator {
   static constexpr size_t page_size = ((1024*1024)/ SIZE+1);
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


//
//template<size_t SIZE>
//class cuda_malloc_allocator {
//   static constexpr size_t page_size = ((1024*1024)/ SIZE+1);
//   thread_local static std::stack<int8_t*> freelist;
//   static std::stack<int8_t*> globallist;
//   static std::atomic<int> lock;
//public:
//   cuda_malloc_allocator() {
//   }
//   void* allocate() {
//      if (freelist.empty()) {
//         while (lock++ != 0) {
//            lock--;
//         }
//         if (globallist.size() < page_size) {
//            lock--;
//            int8_t *ptr;
//            CUDA_CHECK(cudaMalloc(ptr, page_size * SIZE));
//            for (int i = 0; i < page_size; i++) {
//               freelist.push(ptr + i * SIZE);
//            }
//         } else {
//            for (int i = 0; i < page_size; i++) {
//               freelist.push(globallist.top());
//               globallist.pop();
//            }
//            lock--;
//         }
//      }
//      int8_t *ptr = freelist.top();
//      freelist.pop();
//      return ptr;
//   }
//   ~cuda_malloc_allocator() {
//   }
//   void deallocate(void *ptr) {
//      freelist.push((int8_t*) ptr);
//      if (freelist.size() >= 2 * page_size) {
//         while (lock++ != 0) {
//            lock--;
//         }
//         for (int i = 0; i < page_size; i++) {
//            globallist.push(freelist.top());
//            freelist.pop();
//         }
//         lock--;
//      }
//   }
//};
//
//template<size_t SIZE>
//std::atomic<int> cuda_malloc_allocator<SIZE>::lock(0);
//
//template<size_t SIZE>
//thread_local std::stack<int8_t*> cuda_malloc_allocator<SIZE>::freelist;
//
//template<size_t SIZE>
//std::stack<int8_t*> cuda_malloc_allocator<SIZE>::globallist;
