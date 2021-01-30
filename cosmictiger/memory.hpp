/*
 * MEM.hpp
 *
 *  Created on: Jan 24, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_MEM_HPP_
#define COSMICTIGER_MEM_HPP_

#include <cosmictiger/defs.hpp>

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
   std::vector<T*> allocs;
   T *current_alloc;
   int current_index;
public:
   managed_allocator() {
      CUDA_MALLOC(current_alloc, page_size);
      current_index = 0;
      allocs.push_back(current_alloc);
   }
   ~managed_allocator() {
      for( int i = 0; i < page_size; i++) {
         CUDA_FREE(allocs[i]);
      }
   }
   T* allocate() {
      if( current_index == page_size) {
         CUDA_MALLOC(current_alloc, page_size);
         current_index = 0;
         allocs.push_back(current_alloc);
      }
      return current_alloc + current_index++;
   }

};

#endif /* COSMICTIGER_MEM_HPP_ */
