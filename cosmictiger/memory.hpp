/*
 * MEM.hpp
 *
 *  Created on: Jan 24, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_MEM_HPP_
#define COSMICTIGER_MEM_HPP_

#include <cosmictiger/defs.hpp>
#include <cosmictiger/hpx.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_POINTER(ptr)   MEM_CHECK_POINTER(ptr,__FILE__,__LINE__)

#define MEM_CHECK_POINTER(ptr,file,line)                          \
   if( !ptr ) {                                                   \
      printf( "Out of memory. File: %s Line %i\n", file, line);   \
      abort();                                                    \
   }

#define MEM_CHECK_ERROR(ec,file,line)                                                        \
   if( ec != cudaSuccess ) {                                                                 \
      printf( "CUDA error \"%s\" File: %s Line: %i\n",  cudaGetErrorString(ec), file, line); \
      abort();                                                                               \
   }

#define CUDA_FREE(ptr)                                                                                       \
      if( ptr == nullptr ) {                                                                                 \
         printf( "Attempt to free null pointer. File: %s Line %i\n", __FILE__, __LINE__);                    \
         abort();                                                                                            \
      } else {                                                                                               \
         const auto ec = cudaFree(ptr);                                                                      \
         if( ec != cudaSuccess ) {                                                                           \
            printf( "CUDA error \"%s\" File: %s Line: %i\n",  cudaGetErrorString(ec), __FILE__, __LINE__);   \
            abort();                                                                                         \
         }                                                                                                   \
      }

#define CUDA_MALLOC(ptr,nele) cuda_malloc(&ptr,nele,__FILE__,__LINE__)

#define MALLOC(ptr,nele) cosmic_malloc(&ptr,nele,__FILE__,__LINE__)
#define FREE(ptr) free(ptr)

template<class T>
void cuda_malloc(T **ptr, int64_t nele, const char *file, int line) {
   const auto ec = cudaMallocManaged(ptr, nele * sizeof(T));
   MEM_CHECK_POINTER(*ptr, file, line);
   MEM_CHECK_ERROR(ec, file, line);
}

template<class T>
void cosmic_malloc(T **ptr, int64_t nele, const char *file, int line) {
   *ptr = (T*) malloc(nele * sizeof(T));
   MEM_CHECK_POINTER(*ptr, file, line);
}

template<class T>
class managed_allocator {
   static constexpr size_t page_size = ALLOCATION_PAGE_SIZE / sizeof(T);
   static hpx::lcos::local::mutex mtx;
   static std::vector<T*> allocs;
   T *current_alloc;
   int current_index;
public:
   static void cleanup() {
      for (int i = 0; i < allocs.size(); i++) {
         CUDA_FREE(allocs[i]);
      }
      allocs = decltype(allocs)();
   }
   managed_allocator() {
      CUDA_MALLOC(current_alloc, page_size);
      current_index = page_size;
   }
   T* allocate() {
      if (current_index == page_size) {
         CUDA_MALLOC(current_alloc, page_size);
         current_index = 0;
         std::lock_guard < hpx::lcos::local::mutex > lock(mtx);
         allocs.push_back(current_alloc);
      }
      return current_alloc + current_index++;
   }
   managed_allocator( managed_allocator&&) = default;
   managed_allocator& operator=( managed_allocator&&) = default;
   managed_allocator(const managed_allocator&) = delete;
   managed_allocator& operator=(const managed_allocator&) = delete;
};

template<class T>
hpx::lcos::local::mutex managed_allocator<T>::mtx;

template<class T>
std::vector<T*> managed_allocator<T>::allocs;

#endif /* COSMICTIGER_MEM_HPP_ */
