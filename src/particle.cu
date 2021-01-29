#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <cosmictiger/particle.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/memory.hpp>
#include <cosmictiger/fixed.hpp>

#define BLOCK_SIZE 32
#define COUNT_BLOCKS 92

CUDA_KERNEL morton_keygen(particle::flags_t *flags, morton_t *keys_min, morton_t *keys_max, fixed32 *xptr, fixed32 *yptr,
      fixed32 *zptr, size_t nele, size_t depth) {
   const int &tid = threadIdx.x;
   const int &bid = blockIdx.x;
   const size_t shift = (sizeof(fixed32) * CHAR_BIT - depth / NDIM);
   const size_t start = bid * nele / gridDim.x;
   const size_t stop = (bid + 1) * nele / gridDim.x;
   __shared__ morton_t
   maxes[BLOCK_SIZE];
   __shared__ morton_t
   mines[BLOCK_SIZE];
   mines[tid] = ~(size_t(1) << (depth + 1));
   maxes[tid] = 0;
   for (size_t i = start + tid; i < stop; i += BLOCK_SIZE) {
      morton_t key = 0LL;
      size_t x[NDIM];
      x[0] = xptr[i].get_integer() >> shift;
      x[1] = yptr[i].get_integer() >> shift;
      x[2] = zptr[i].get_integer() >> shift;
      for (size_t k = 0; k < depth / NDIM; k++) {
         for (size_t dim = 0; dim < NDIM; dim++) {
            key ^= size_t((bool) (x[dim] & (1LL << k))) << size_t(k * NDIM + (NDIM-dim-1));
         }
      }
      //     printf( "%lx\n",key);
      maxes[tid] = max(maxes[tid], key);
      mines[tid] = min(mines[tid], key);
      flags[i].morton_id = key;
      //   printf( "%i\n", nele);
   }
   __syncthreads();
   for (int P = BLOCK_SIZE / 2; P >= 1; P /= 2) {
      if (tid < P) {
         maxes[tid] = max(maxes[tid], maxes[tid + P]);
         mines[tid] = min(mines[tid], mines[tid + P]);
      }
      __syncthreads();
   }
   if( tid == 0 ) {
      keys_min[blockIdx.x] = mines[0];
      keys_max[blockIdx.x] = maxes[0];
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

std::vector<size_t> cuda_keygen(particle_set &set, size_t start, size_t stop, int depth, morton_t &kmin,
      morton_t &kmax) {
   morton_t key_start = kmin;
   morton_t key_stop = kmax;
   morton_t *key_min;
   morton_t *key_max;
   const int nblocks = (92 * 32 - 1) / BLOCK_SIZE + 1;
   CUDA_MALLOC(key_min, nblocks);
   CUDA_MALLOC(key_max, nblocks);
   size_t total_keys = (1 << (depth + 1));
   *key_min = ~total_keys;
   *key_max = 0;
   start -= set.offset_;
   stop -= set.offset_;
   assert( stop > start);
   fixed32 *x = set.xptr_[0] + start;
   fixed32 *y = set.xptr_[1] + start;
   fixed32 *z = set.xptr_[2] + start;
   particle::flags_t *flags = set.rptr_ + start;
morton_keygen<<<nblocks, BLOCK_SIZE>>>(flags,key_min,key_max,x,y,z,stop-start, depth);
            CUDA_CHECK(cudaDeviceSynchronize());
   int *counts;

   for( int i = 1; i < nblocks; i++) {
      *key_min = std::min(*key_min,key_min[i]);
      *key_max = std::max(*key_max,key_max[i]);
     }
   (*key_max)++;
//   printf( "KEYS         %lx %lx \n", *key_min, *key_max);

   assert(*key_max - *key_min <= key_stop - key_start);
//   if( *key_min < key_start || *key_max > key_stop) {
//      printf( "Key out of range\n");
//      printf( "%li %li %li %li\n", key_start, *key_min, *key_max, key_stop) ;
   //    abort();
 //  }
   const size_t size = *key_max - *key_min;
   CUDA_MALLOC(counts, size);
   for (int i = 0; i < size; i++) {
      counts[i] = 0;
   }
   assert(key_stop - key_start + 1  >= *key_max - *key_min);
   count_keys<<<COUNT_BLOCKS,BLOCK_SIZE>>>(counts,  flags, *key_min, *key_max, stop - start);
            CUDA_CHECK(cudaDeviceSynchronize());
//            printf( "%li %li %li %li  %li  \n", key_start, *key_min, *key_max, key_stop, key_stop - key_start + 1 >= *key_max - *key_min) ;
         std::vector < size_t > bounds(key_stop - key_start + 1);
//         printf( "%li %li %li %li  %li  \n", key_start, *key_min, *key_max, key_stop, key_stop - key_start + 1 >= *key_max - *key_min) ;
   assert(key_stop - key_start + 1  >= *key_max - *key_min);
   bounds[0] = start + set.offset_;
   for (size_t i = 1; i <= *key_max- *key_min; i++) {
      bounds[i] = bounds[i - 1] + counts[i - 1];
   }

   kmin = *key_min;
   kmax = *key_max;
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
