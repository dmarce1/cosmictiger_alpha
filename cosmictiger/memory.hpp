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
#include <cosmictiger/cuda.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <stack>
#include <unordered_map>

#define CHECK_POINTER(ptr)   MEM_CHECK_POINTER(ptr,__FILE__,__LINE__)

#define MEM_CHECK_POINTER(ptr,file,line)                          \
   if( !ptr ) {                                                   \
      printf( "Out of memory. File: %s Line %i\n", file, line);   \
      ABORT();                                                    \
   }

#define MEM_CHECK_ERROR(ec,file,line)                                                        \
   if( ec != cudaSuccess ) {                                                                 \
      printf( "CUDA error \"%s\" File: %s Line: %i\n",  cudaGetErrorString(ec), file, line); \
      ABORT();                                                                               \
   }

#ifdef __CUDA_ARCH__
#define CUDA_FREE(ptr)                                                                                       \
      if( ptr == nullptr ) {                                                                                 \
         printf( "Attempt to free null pointer. File: %s Line %i\n", __FILE__, __LINE__);                    \
         ABORT();                                                                                            \
      } else {                                                                                               \
         free(ptr);                                                                      \
      }
#else
#define CUDA_FREE(ptr)                                                                                       \
      if( ptr == nullptr ) {                                                                                 \
         printf( "Attempt to free null pointer. File: %s Line %i\n", __FILE__, __LINE__);                    \
         ABORT();                                                                                            \
      } else {                                                                                               \
         const auto ec = cudaFree(ptr);                                                                      \
         if( ec != cudaSuccess ) {                                                                           \
            printf( "CUDA error \"%s\" File: %s Line: %i\n",  cudaGetErrorString(ec), __FILE__, __LINE__);   \
            ABORT();                                                                                         \
         }                                                                                                   \
      }
#endif

#define CUDA_MALLOC(ptr,nele) cuda_malloc(&ptr,nele,__FILE__,__LINE__)

#define MALLOC(ptr,nele) cosmic_malloc(&ptr,nele,__FILE__,__LINE__)
#define FREE(ptr) free(ptr)

template<class T>
CUDA_EXPORT inline void cuda_malloc(T **ptr, int64_t nele, const char *file, int line) {
#ifdef __CUDA_ARCH__
	*ptr = (T*) malloc(nele * sizeof(T));
	MEM_CHECK_POINTER(*ptr, file, line);
#else
	// printf( "Callc\n");
	const auto ec = cudaMallocManaged(ptr, nele * sizeof(T));
	MEM_CHECK_POINTER(*ptr, file, line);
	MEM_CHECK_ERROR(ec, file, line);
#endif
}

template<class T>
void cosmic_malloc(T **ptr, int64_t nele, const char *file, int line) {
	*ptr = (T*) malloc(nele * sizeof(T));
	MEM_CHECK_POINTER(*ptr, file, line);
}

class cuda_allocator {
private:
	static std::vector<std::stack<void*>> freelist;
	static size_t allocated;
	static std::unordered_map<void*, int> delete_indexes;
#ifndef __CUDACC__
	static mutex_type mtx;
#endif
public:
	void* allocate(size_t sz);
	void deallocate(void *ptr);
};

class unified_allocator {
private:
	static std::vector<std::stack<void*>> freelist;
	static size_t allocated;
	static std::unordered_map<void*, int> delete_indexes;
#ifndef __CUDACC__
	static mutex_type mtx;
#endif
public:
	void* allocate(size_t sz);
	void deallocate(void *ptr);
};
cudaStream_t get_stream();
void cleanup_stream(cudaStream_t s);

#ifndef __CUDACC__

template<class T>
class managed_allocator {
	static constexpr size_t page_size = ALLOCATION_PAGE_SIZE / sizeof(T);
	static mutex_type mtx;
	static std::vector<T*> allocs;
	T *current_alloc;
	int current_index;
public:
	static void cleanup() {
		for (int i = 0; i < allocs.size(); i++) {
			allocs[i]->T::~T();
			unified_allocator ua;
			ua.deallocate(allocs[i]);
		}
		allocs = decltype(allocs)();
	}
	managed_allocator() {
		//   printf( "manaed %li\n", sizeof(T));
		current_index = page_size;
	}
	T* allocate() {
		//    printf( "----\n");
		//    printf( "!!!!\n");
		assert(current_index <= page_size);
		if (current_index == page_size) {
			unified_allocator ua;
			current_alloc = (T*) ua.allocate(page_size * sizeof(T));
			current_index = 0;
			std::lock_guard<mutex_type> lock(mtx);
			allocs.push_back(current_alloc);
		}
		new (current_alloc + current_index) T();
		return current_alloc + current_index++;
	}
	static void set_device(int device) {
		auto stream = get_stream();
		printf("Sending trees to GPU %li\n", allocs.size());
		for (auto& alloc : allocs) {		//
			CUDA_CHECK(cudaMemAdvise(alloc, sizeof(T) * page_size, cudaMemAdviseSetPreferredLocation, device));
			CUDA_CHECK(cudaMemAdvise(alloc, sizeof(T) * page_size, cudaMemAdviseSetAccessedBy, device));
			CUDA_CHECK(cudaMemPrefetchAsync(alloc, sizeof(T) * page_size, device, stream));
		}
		CUDA_CHECK(cudaStreamSynchronize(stream));
		printf("Done sending trees to GPU\n");
		cleanup_stream(stream);
	}
	managed_allocator(managed_allocator&&) = default;
	managed_allocator& operator=(managed_allocator&&) = default;
	managed_allocator(const managed_allocator&) = delete;
	managed_allocator& operator=(const managed_allocator&) = delete;
};

template<class T>
mutex_type managed_allocator<T>::mtx;

template<class T>
std::vector<T*> managed_allocator<T>::allocs;

#endif

#endif /* COSMICTIGER_MEM_HPP_ */
