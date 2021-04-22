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

static size_t allocated = 0;
static std::unordered_map<void*, size_t> alloc_map;
static std::unordered_map<void*, int> linenumbers;
static std::unordered_map<void*, std::string> filenames;
static mutex_type mtx;
#include <sys/sysinfo.h>

size_t cuda_unified_total() {
	return allocated;
}

void* cuda_unified_alloc(size_t sz, const char* file, int line) {
	char* ptr;
	CUDA_CHECK(cudaMallocManaged(&ptr, sz));
	if (ptr == nullptr) {
		printf("Unable to allocated unified memory\n");
		abort();
	}
//	std::lock_guard<mutex_type> lock(mtx);
	alloc_map[ptr] = sz;
	allocated += sz;
//	linenumbers[ptr] = line;
//	filenames[ptr] = std::string(file);
	//printf("%li KB  allocated %e total GB allocated %s %i\n", sz/1024, allocated / 1024.0 / 1024 / 1024, file, line);
	return ptr;
}

void cuda_unified_show_outstanding() {
	/*	for (auto i = alloc_map.begin(); i != alloc_map.end(); i++) {
	 printf("%li KB %s %i\n", i->second / 1024, filenames[i->first].c_str(), linenumbers[i->first]);
	 }*/
}

void cuda_unified_free(void* ptr) {
	CUDA_CHECK(cudaFree(ptr));
	//std::lock_guard<mutex_type> lock(mtx);
	allocated -= alloc_map[ptr];
	//filenames.erase(ptr);
	//linenumbers.erase(ptr);
	alloc_map.erase(ptr);
}

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
	return ptr;
}

void cuda_allocator::deallocate(void *ptr) {
	std::lock_guard<mutex_type> lock(mtx);
	const auto index = delete_indexes[ptr];
	freelist[index].push(ptr);
}

std::vector<std::stack<void*>> cuda_allocator::freelist;
mutex_type cuda_allocator::mtx;

std::unordered_map<void*, int> cuda_allocator::delete_indexes;
std::stack<void*> unified_allocator::allocs;

void unified_allocator::reset() {
	if (allocated != 0) {
		printf("Attempt to reset unified allocator without freeing all memory, %li left allocated.\n", allocated);
		abort();
	}
	printf("%e GB BEFORE alloc reset\n", ::allocated / 1024.0 / 1024 / 1024);
	freelist = decltype(freelist)();
	while (allocs.size()) {
		CUDA_FREE(allocs.top());
		allocs.pop();
	}
	printf("%e GB AFTER alloc reset\n", ::allocated / 1024.0 / 1024 / 1024);
	cuda_unified_show_outstanding();
}

void* unified_allocator::allocate(size_t sz) {
	/*uint8_t* ptr;
	 CUDA_MALLOC(ptr,sz);
	 return ptr;*/

	size_t chunk_size = 1;
	int index = 0;
	size_t total_sz = sizeof(size_t) + sz;
	while (chunk_size < total_sz) {
		chunk_size *= 2;
		index++;
	}
	void *ptr;
	char* cptr;
	size_t nchunks = std::max(16 * 1024 * 1024 / (size_t) chunk_size, (size_t) 1);
	std::lock_guard<mutex_type> lock(mtx);
	freelist.resize(std::max((int) freelist.size(), index + 1));
	if (freelist[index].empty()) {
		CUDA_MALLOC(cptr, nchunks * chunk_size);
		allocs.push(cptr);
		ptr = cptr;
		for (int i = 0; i < nchunks; i++) {
			size_t* head_ptr = (size_t*) ((char*) ptr + i * chunk_size);
			void* data_ptr = (char*) head_ptr + sizeof(size_t);
			*head_ptr = index;
			freelist[index].push(data_ptr);
		}
	}
	ptr = freelist[index].top();
	freelist[index].pop();
	allocated += (1 << index);
	return ptr;
}

void unified_allocator::deallocate(void *ptr) {
	size_t* head_ptr = (size_t*)((char*)ptr - sizeof(size_t));
	const size_t index = *head_ptr;
	std::lock_guard<mutex_type> lock(mtx);
	allocated -= (1 << index);
	freelist[index].push(ptr);
}

std::vector<std::stack<void*>> unified_allocator::freelist;
mutex_type unified_allocator::mtx;

size_t unified_allocator::allocated = 0;

