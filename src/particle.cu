#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <cosmictiger/particle.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/memory.hpp>
#include <cosmictiger/fixed.hpp>

#define BLOCK_SIZE 32
#define COUNT_BLOCKS 92

CUDA_KERNEL morton_keygen(particle::flags_t *flags, morton_t *key_min, morton_t *key_max, fixed32 *xptr, fixed32 *yptr,
      fixed32 *zptr, size_t nele, size_t depth) {
   const int &tid = threadIdx.x + blockDim.x * BLOCK_SIZE;
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
            key ^= size_t((bool) (x[dim] & (0x0000000000000001LL << k))) << size_t(k * NDIM + dim);
         }
      }
      //     printf( "%lx\n",key);
      *key_min = atomicMin((unsigned long long*) key_min, (unsigned long long) key);
      *key_max = atomicMax((unsigned long long*) key_max, (unsigned long long) key);
      flags[i].morton_id = key;
      //   printf( "%i\n", nele);
      __syncthreads();
   }
}

CUDA_KERNEL count_keys(int *counts, particle::flags_t *keys, morton_t key_min, morton_t key_max, size_t nele) {
   const int &tid = threadIdx.x + blockDim.x * BLOCK_SIZE;
   const int &bid = blockIdx.x;
   const size_t start = bid * nele / gridDim.x;
   const size_t stop = (bid + 1) * nele / gridDim.x;
   for (size_t i = start + tid; i < stop; i += BLOCK_SIZE) {
      const size_t index = keys[i].morton_id - key_min;
      atomicAdd(counts + 0, 1);
   }
}

std::vector<size_t> cuda_keygen(particle_set &set, size_t start, size_t stop, int depth) {
   morton_t *key_min;
   morton_t *key_max;
   CUDA_MALLOC(key_min, 1);
   CUDA_MALLOC(key_max, 1);
   *key_min = ~(1 << (depth + 1));
   *key_max = 0;
   start -= set.offset_;
   stop -= set.offset_;
   fixed32 *x = set.xptr_[0] + start;
   fixed32 *y = set.xptr_[1] + start;
   fixed32 *z = set.xptr_[2] + start;
   particle::flags_t *flags = set.rptr_ + start;
//
   const int nblocks = (92 * 32 - 1) / BLOCK_SIZE + 1;
morton_keygen<<<nblocks, BLOCK_SIZE>>>(flags,key_min,key_max,x,y,z,stop-start, depth);
         CUDA_CHECK(cudaDeviceSynchronize());
  //
       //  printf("-----%li %li\n", *key_min, *key_max);
   int *counts;
   (*key_max)++;
   const size_t size = *key_max - *key_min;
 //  printf( "-------%li\n", size);
   CUDA_MALLOC(counts, size);
   for (int i = 0; i < *key_max - *key_min; i++) {
      counts[i] = 0;
   }
count_keys<<<COUNT_BLOCKS,BLOCK_SIZE>>>(counts,  flags, *key_min, *key_max, stop-start);
         CUDA_CHECK(cudaDeviceSynchronize());
   std::vector < size_t > bounds(*key_max - *key_min);
   bounds[0] = start + set.offset_;
   for (size_t i = 1; i < *key_max - *key_min; i++) {
      bounds[i] = bounds[i - 1] + counts[i - 1];
   }

   CUDA_FREE(counts);
   CUDA_FREE(key_max);
   CUDA_FREE(key_min);

   return bounds;
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
