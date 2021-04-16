/*
 * memory.cpp
 *
 *  Created on: Feb 12, 2021
 *      Author: dmarce1
 */

#include <cosmictiger/cuda.hpp>

#include <vector>
#include <stack>
#include <unordered_map>

#include <cosmictiger/memory.hpp>

void* cuda_allocator::allocate(size_t sz) {
	int total_sz = 1;
	int index = 0;
	while (total_sz < sz) {
		total_sz *= 2;
		index++;
	}
	int chunk_size = std::max(std::min(32, 2 * 1024 * 1024 / (int) total_sz), 1);
	// printf( "%li\n", sz);
	std::lock_guard<mutex_type> lock(mtx);
	freelist.resize(std::max((int) freelist.size(), index + 1));
	void *ptr;
//   printf( "%li %i\n", freelist[index].size(), index);
	if (freelist[index].empty()) {

		//   	printf( "Allocating %li bytes on device\n", chunk_size * total_sz);
		CUDA_CHECK(cudaMalloc(&ptr, chunk_size * total_sz));
//      printf( "Done Allocating %li bytes on device\n", chunk_size * total_sz);
		for (int i = 0; i < chunk_size; i++) {
			freelist[index].push((char*) ptr + i * total_sz);
		}
	}
	ptr = freelist[index].top();
	delete_indexes[ptr] = index;
	freelist[index].pop();
//  printf( "%li\n", (int) allocated);
	return ptr;
}

void cuda_allocator::deallocate(void *ptr) {
	std::lock_guard<mutex_type> lock(mtx);
	const auto index = delete_indexes[ptr];
	freelist[index].push(ptr);
//   printf( "%li\n", (int) allocated);
}

std::vector<std::stack<void*>> cuda_allocator::freelist;
mutex_type cuda_allocator::mtx;
size_t cuda_allocator::allocated = 0;

std::unordered_map<void*, int> cuda_allocator::delete_indexes;
std::unordered_map<int, int> unified_allocator::alloc_counts;

void unified_allocator::show_allocs() {
	printf("Allocations:\n");
	for (auto i = alloc_counts.begin(); i != alloc_counts.end(); i++) {
		if (i->second) {
			printf("%i %i\n", i->second, i->first);
		}
	}
	printf("%li MB in use\n", allocated / 1024 / 1024);
}

void* unified_allocator::allocate(size_t sz) {
	/*uint8_t* ptr;
	 CUDA_MALLOC(ptr,sz);
	 return ptr;*/

	size_t total_sz = 1;
	int index = 0;
	while (total_sz < sz) {
		total_sz *= 2;
		index++;
	}
	std::lock_guard<mutex_type> lock(mtx);
	if (alloc_counts.find(total_sz) == alloc_counts.end()) {
		alloc_counts[total_sz] = 0;
	}
	size_t chunk_size = std::max(std::min((size_t)1024, 64 * 1024 * 1024 / (size_t) total_sz), (size_t) 1);
	freelist.resize(std::max((int) freelist.size(), index + 1));
	void *ptr;
	char* cptr;
	if (freelist[index].empty()) {
		printf("Allocating %li bytes UNIFIED\n", total_sz);
		CUDA_MALLOC(cptr, chunk_size * total_sz);
		ptr = cptr;
		for (int i = 0; i < chunk_size; i++) {
			freelist[index].push((char*) ptr + i * total_sz);
		}
		allocated += chunk_size * total_sz;
	}
	ptr = freelist[index].top();
	delete_indexes[ptr] = index;
	freelist[index].pop();
	alloc_counts[total_sz]++;
	return ptr;
}

void unified_allocator::deallocate(void *ptr) {
	//CUDA_FREE(ptr);
	std::lock_guard<mutex_type> lock(mtx);
	if (delete_indexes.find(ptr) == delete_indexes.end()) {
		printf("attempted to free invalid unified pointer\n");
		abort();
	}
	const auto index = delete_indexes[ptr];
	alloc_counts[1 << index]--;
	freelist[index].push(ptr);
}

std::vector<std::stack<void*>> unified_allocator::freelist;
mutex_type unified_allocator::mtx;
size_t unified_allocator::allocated = 0;

std::unordered_map<void*, int> unified_allocator::delete_indexes;

