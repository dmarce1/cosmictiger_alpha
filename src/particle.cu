#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <cosmictiger/particle.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/memory.hpp>
#include <cosmictiger/fixed.hpp>

#define BLOCK_SIZE 32
#define COUNT_BLOCKS 92

CUDA_KERNEL morton_keygen(particle::flags_t *flags, fixed32 *xptr, fixed32 *yptr, fixed32 *zptr, size_t nele,
      size_t depth) {
   const int &tid = threadIdx.x;
   const int &bid = blockIdx.x;
   const size_t shift = (sizeof(fixed32) * CHAR_BIT - depth / NDIM);
   const size_t start = bid * nele / gridDim.x;
   const size_t stop = (bid + 1) * nele / gridDim.x;
   for (size_t i = start + tid; i < stop; i += BLOCK_SIZE) {
      morton_t key = 0LL;
      size_t x[NDIM];
      x[0] = xptr[i].get_integer() >> shift;
      x[1] = yptr[i].get_integer() >> shift;
      x[2] = zptr[i].get_integer() >> shift;
      for (size_t k = 0; k < depth / NDIM; k++) {
         for (size_t dim = 0; dim < NDIM; dim++) {
            key ^= size_t((bool) (x[dim] & (1LL << k))) << size_t(k * NDIM + (NDIM - 1 - dim));
         }
      }
      //     printf( "%lx\n",key);
      flags[i].morton_id = key;
      //   printf( "%i\n", nele);
   }
}

CUDA_KERNEL count_keys(int *counts, particle::flags_t *keys, morton_t key_min, morton_t key_max, size_t nele) {
   const int &tid = threadIdx.x;
   const int &bid = blockIdx.x;
   const size_t start = bid * nele / gridDim.x;
   const size_t stop = (bid + 1) * nele / gridDim.x;
   for (size_t i = start + tid; i < stop; i += BLOCK_SIZE) {
      const size_t index = keys[i].morton_id - key_min;
//      if(keys[i].morton_id < key_min ) {
//         printf( "min out %lx %lx\n", keys[i].morton_id, key_min);
//         __trap();
//      }
//      if(keys[i].morton_id >= key_max ) {
//         printf( "max out %lx %lx\n", keys[i].morton_id, key_max);
//         __trap();
//      }
      assert(keys[i].morton_id >= key_min);
      assert(keys[i].morton_id < key_max);
      atomicAdd(counts + index, 1);
   }
}

std::pair<std::vector<size_t>, std::vector<size_t>> cuda_keygen(particle_set &set, size_t start, size_t stop, int depth,
      morton_t key_min, morton_t key_max) {
   const int nblocks = (92 * 32 - 1) / BLOCK_SIZE + 1;
   int *counts;
   const size_t size = key_max - key_min;
   start -= set.offset_;
   stop -= set.offset_;
   assert(stop > start);
   fixed32 *x = set.xptr_[0] + start;
   fixed32 *y = set.xptr_[1] + start;
   fixed32 *z = set.xptr_[2] + start;
   particle::flags_t *flags = set.rptr_ + start;
morton_keygen<<<nblocks, BLOCK_SIZE>>>(flags,x,y,z,stop-start, depth);
                     CUDA_CHECK(cudaDeviceSynchronize());
   // printf( "KEYS         %lx %lx %lx %lx \n", key_min, *key_min, *key_max, key_stop);

   CUDA_MALLOC(counts, size);
   for (int i = 0; i < size; i++) {
      counts[i] = 0;
   }
count_keys<<<COUNT_BLOCKS,BLOCK_SIZE>>>(counts,  flags, key_min, key_max, stop - start);
                     CUDA_CHECK(cudaDeviceSynchronize());
//            printf( "%li %li %li %li  %li  \n", key_min, *key_min, *key_max, key_stop, key_stop - key_min + 1 >= *key_max - *key_min) ;

   std::pair<std::vector<size_t>, std::vector<size_t>> bounds;
   bounds.first.resize(key_max - key_min + 1);
   bounds.second.resize(key_max - key_min + 1);

   bounds.first[0] = start;
   for (int i = 1; i <= size; i++) {
      bounds.first[i] = counts[i - 1] + bounds.first[i - 1];
   }
   CUDA_FREE(counts);
   for (int i = 1; i <= size; i++) {
      bounds.second[i - 1] = bounds.first[i];
   }
   return std::move(bounds);
}

//
//CUDA_KERNEL radix_sort_count(size_t *count, morton_t *keys, morton_t key_min, morton_t key_max) {
//
//}
//
//
//CUDA_KERNEL radix_sort_do_sort(fixed32 *x, fixed32 *y, fixed32 *z, fixed32 *vx, fixed32 *vy, fixed32 *vz, rung_t *rung,
//      size_t *begin, size_t *end, morton_t key_min, size_t nele) {
//
//}
