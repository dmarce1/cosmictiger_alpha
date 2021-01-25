/*
 * MEM.hpp
 *
 *  Created on: Jan 24, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_MEM_HPP_
#define COSMICTIGER_MEM_HPP_

#include <cuda_runtime.h>

#define MEM_CHECK_POINTER(ptr,file,line)                          \
	if( !ptr ) {                            						      \
		printf( "Out of MEM. File: %s Line %i\n", file, line); 	   \
		abort();													               \
	}

#define MEM_CHECK_ERROR(ec,file,line)                                                        \
   if( ec != cudaSuccess ) {                                                                 \
      printf( "CUDA error \"%s\" File: %s Line: %i\n",  cudaGetErrorString(ec), file, line); \
      abort();                                                                               \
   }

#define CUDA_FREE(ptr)                                                                      \
      if( ptr == nullptr ) {                                                                         \
         printf( "Attempt to free null pointer. File: %s Line %i\n", __FILE__, __LINE__);              \
         abort();                                                                                    \
      } else {                                                                                       \
         const auto ec = cudaFree(ptr);                                                              \
         if( ec != cudaSuccess ) {                                                                   \
            printf( "CUDA error \"%s\" File: %s Line: %i\n",  cudaGetErrorString(ec), __FILE__, __LINE__);   \
            abort();                                                                                 \
         }                                                                                           \
      }

#define CUDA_MALLOC(ptr,nele) cuda_malloc(&ptr,nele,__FILE__,__LINE__)

template<class T>
void cuda_malloc(T **ptr, int64_t nele, const char *file, int line) {
   const auto ec = cudaMallocManaged(ptr, nele * sizeof(T));
   MEM_CHECK_POINTER(*ptr, file, line);
   MEM_CHECK_ERROR(ec, file, line);
}

#endif /* COSMICTIGER_MEM_HPP_ */
