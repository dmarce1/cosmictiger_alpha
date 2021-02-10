#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <cosmictiger/particle.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/memory.hpp>
#include <cosmictiger/fixed.hpp>

#define THREADS_PER_SM 64
#define KEY_BLOCK_SIZE 256
#define COUNT_BLOCK_SIZE 256

CUDA_KERNEL morton_keygen(particle::flags_t *flags, fixed32 *xptr, fixed32 *yptr, fixed32 *zptr, size_t nele,
      size_t depth) {
   const int &tid = threadIdx.x;
   const int &bid = blockIdx.x;
   const size_t start = bid * nele / gridDim.x;
   const size_t stop = (bid + 1) * nele / gridDim.x;
   for (size_t i = start + tid; i < stop; i += KEY_BLOCK_SIZE) {
      flags[i].morton_id = morton_key(xptr[i], yptr[i], zptr[i], depth);
   }
}

CUDA_KERNEL count_keys(int *counts, particle::flags_t *keys, morton_t key_min, morton_t key_max, size_t nele) {
   const int &tid = threadIdx.x;
   const int &bid = blockIdx.x;
   const size_t start = bid * nele / gridDim.x;
   const size_t stop = (bid + 1) * nele / gridDim.x;
   for (size_t i = start + tid; i < stop; i += COUNT_BLOCK_SIZE) {
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

std::vector<size_t> cuda_keygen(particle_set &set, size_t start, size_t stop, int depth, morton_t key_start,
      morton_t key_stop) {
   const int nprocs = global().cuda.devices[0].multiProcessorCount;
   const int nkeyblocks = (nprocs * THREADS_PER_SM - 1) / KEY_BLOCK_SIZE + 1;
   const int ncountblocks = (nprocs * THREADS_PER_SM - 1) / COUNT_BLOCK_SIZE + 1;
   start -= set.offset_;
   stop -= set.offset_;
   assert(stop > start);
   fixed32 *x = set.xptr_[0] + start;
   fixed32 *y = set.xptr_[1] + start;
   fixed32 *z = set.xptr_[2] + start;
   particle::flags_t *flags = set.rptr_ + start;
   /***********************************************************************************/
   /**/morton_keygen<<<nkeyblocks, KEY_BLOCK_SIZE>>>(flags,x,y,z,stop-start, depth);/**/
   /**/CUDA_CHECK(cudaDeviceSynchronize()); /**/
   /***********************************************************************************/

   int *counts;
   const size_t size = key_stop - key_start;
   CUDA_MALLOC(counts, size);
   for (int i = 0; i < size; i++) {
      counts[i] = 0;
   }
   assert(key_stop - key_start + 1 >= key_stop - key_start);
   /*******************************************************************************************************/
   /**/count_keys<<<ncountblocks,COUNT_BLOCK_SIZE>>>(counts,  flags, key_start, key_stop, stop - start);/**/
   /**/CUDA_CHECK(cudaDeviceSynchronize()); /**/
   /*******************************************************************************************************/
   std::vector < size_t > bounds(key_stop - key_start + 1);
   assert(key_stop - key_start + 1 >= key_stop - key_start);
   bounds[0] = start + set.offset_;
   for (size_t i = 1; i <= key_stop - key_start; i++) {
      bounds[i] = bounds[i - 1] + counts[i - 1];
   }

   CUDA_FREE(counts);

   return bounds;
}

CUDA_KERNEL drift_kernel(particle_set *parts, double a1, double a2, double dtau);


void drift(particle_set* parts, double a1, double a2, double dtau) {
   drift_kernel<<<1024,32>>>(parts,a1,a2,dtau);
   CUDA_CHECK(cudaDeviceSynchronize());
}

CUDA_KERNEL drift_kernel(particle_set *parts, double a1, double a2, double dtau) {
   const int &tid = threadIdx.x;
   const int &bid = blockIdx.x;
   const int &bsz = blockDim.x;
   const int &gsz = gridDim.x;
   const size_t start = bid * parts->size() / gsz;
   const size_t stop = (bid + 1) * parts->size() / gsz;
   const auto dt = (0.5 / a1 + 0.5 / a2)*dtau;
  for (size_t i = start + tid; i < stop; i += bsz) {
      double x = parts->pos(0,i).to_double();
      double y = parts->pos(1,i).to_double();
      double z = parts->pos(2,i).to_double();
      const double vx = parts->vel(0,i);
      const double vy = parts->vel(1,i);
      const double vz = parts->vel(2,i);
      x += vx * dt;
      y += vy * dt;
      z += vz * dt;
  ///    printf( "%e %e %e\n", x, y, z);
 /*     while( x > 1.0 ) {
         x -= 1.0;
      }
      while( y > 1.0 ) {
         y -= 1.0;
      }
      while( z > 1.0 ) {
         z -= 1.0;
      }
      while( x < 0.0 ) {
         x += 1.0;
      }
      while( y < 0.0 ) {
         y += 1.0;
      }
      while( z < 0.0 ) {
         z += 1.0;
      }*/
      parts->pos(0,i) = x;
      parts->pos(1,i) = y;
      parts->pos(2,i) = z;
   }
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
