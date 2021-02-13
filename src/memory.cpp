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
   int total_sz = 1024;
   constexpr int chunk_size = 32;
   int index = 0;
   while (total_sz < sz) {
      total_sz *= 2;
      index++;
   }
   std::lock_guard < mutex_type > lock(mtx);
   freelist.resize(index + 1);
   void *ptr;
   if (freelist[index].empty()) {
      CUDA_CHECK(cudaMalloc(&ptr, chunk_size * total_sz));
      for( int i = 0; i < chunk_size; i++) {
         freelist[index].push(ptr + i * total_sz);
      }
   }
   ptr = freelist[index].top();
   delete_indexes[ptr] = index;
   freelist[index].pop();
   return ptr;
}

void cuda_allocator::deallocate(void *ptr) {
   std::lock_guard < mutex_type > lock(mtx);
   const auto index = delete_indexes[ptr];
   freelist[index].push(ptr);
}

std::vector<std::stack<void*>> cuda_allocator::freelist;
mutex_type cuda_allocator::mtx;

std::unordered_map<void*, int> cuda_allocator::delete_indexes;

