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
   int chunk_size = std::max(std::min(32,2*1024*1024/(int)total_sz),1);
  // printf( "%li\n", sz);
   std::lock_guard < mutex_type > lock(mtx);
   allocated += total_sz;
   freelist.resize(std::max((int) freelist.size(),index + 1));
   void *ptr;
//   printf( "%li %i\n", freelist[index].size(), index);
   if (freelist[index].empty()) {
      CUDA_CHECK(cudaMalloc(&ptr, chunk_size * total_sz));
 //     printf( "Allocating %li bytes on device\n", chunk_size * total_sz);
      for( int i = 0; i < chunk_size; i++) {
         freelist[index].push(ptr + i * total_sz);
      }
   }
   ptr = freelist[index].top();
   delete_indexes[ptr] = index;
   freelist[index].pop();
//  printf( "%li\n", (int) allocated);
   return ptr;
}

void cuda_allocator::deallocate(void *ptr) {
   std::lock_guard < mutex_type > lock(mtx);
   const auto index = delete_indexes[ptr];
   allocated -= (1<<index);
   freelist[index].push(ptr);
//   printf( "%li\n", (int) allocated);
}

std::vector<std::stack<void*>> cuda_allocator::freelist;
mutex_type cuda_allocator::mtx;
size_t cuda_allocator::allocated = 0;

std::unordered_map<void*, int> cuda_allocator::delete_indexes;








void* unified_allocator::allocate(size_t sz) {
   int total_sz = 1;
   int index = 0;
   while (total_sz < sz) {
      total_sz *= 2;
      index++;
   }
   int chunk_size = std::max(std::min(32,2*1024*1024/(int)total_sz),1);
  // printf( "%li\n", sz);
   std::lock_guard < mutex_type > lock(mtx);
   allocated += total_sz;
   freelist.resize(std::max((int) freelist.size(),index + 1));
   void *ptr;
   char* cptr;
//   printf( "%li %i\n", freelist[index].size(), index);
   if (freelist[index].empty()) {
      CUDA_MALLOC(cptr, chunk_size * total_sz);
      ptr = cptr;
  //    printf( "Allocating %li unified bytes on device\n", chunk_size * total_sz);
      for( int i = 0; i < chunk_size; i++) {
         freelist[index].push(ptr + i * total_sz);
      }
   }
   ptr = freelist[index].top();
   delete_indexes[ptr] = index;
   freelist[index].pop();
 // printf( "%li\n", (int) allocated);
   return ptr;
}

void unified_allocator::deallocate(void *ptr) {
   std::lock_guard < mutex_type > lock(mtx);
   const auto index = delete_indexes[ptr];
   allocated -= (1<<index);
   freelist[index].push(ptr);
 //  printf( "%li\n", (int) allocated);
}

std::vector<std::stack<void*>> unified_allocator::freelist;
mutex_type unified_allocator::mtx;
size_t unified_allocator::allocated = 0;

std::unordered_map<void*, int> unified_allocator::delete_indexes;

