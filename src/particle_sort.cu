#include <cosmictiger/defs.hpp>
#include <cosmictiger/particle_sort.hpp>
#include <cosmictiger/cuda.hpp>

struct pointers_t {
   fixed32 *x[NDIM];
   float *v[NDIM];
   int8_t *rung;
};

CUDA_KERNEL particle_sort_kernel(size_t size, pointers_t pointers, int xdim, fixed32 xmid) {
   const size_t tid = threadIdx.x;
   const size_t bid = blockIdx.x;
   const size_t gsz = gridDim.x;
   const size_t start = bid * size / gsz;
   const size_t stop = (bid + (size_t) 1) * size / gsz;
   auto &x = pointers.x;
   auto &v = pointers.v;
   auto &rung = pointers.rung;

   if (tid == 0) {
      size_t hi = stop - 1;
      size_t lo = start;
      while (lo < hi) {
         if (x[xdim][lo] > xmid) {
            while (x[xdim][hi] > xmid && lo < hi) {
               hi--;
            }
            if (lo < hi) {
               for (int dim = 0; dim < NDIM; dim++) {
                  swap(x[dim][lo], x[dim][hi]);
               }
               for (int dim = 0; dim < NDIM; dim++) {
                  const float tmp = v[dim][lo];
                  v[dim][lo] = v[dim][hi];
                  v[dim][hi] = tmp;
               }
               const float tmp = rung[lo];
               rung[lo] = rung[hi];
               rung[hi] = tmp;
            }
         }
         lo++;
      }
   }

   __syncthreads();
}

size_t particle_sort::cuda_sort(size_t begin, size_t end, int xdim, fixed32 xmid) {
   pointers_t ptrs;
   for (int dim = 0; dim < NDIM; dim++) {
      ptrs.x[dim] = parts.x[dim] + begin + parts.offset;
      ptrs.v[dim] = parts.v[dim] + begin + parts.offset;
   }
   ptrs.rung = parts.rung + begin + parts.offset;
   const size_t size = end - begin;
   /**************************************/
   /**/particle_sort_kernel<<<92,32>>>(size,ptrs,xdim,xmid);/***/
/**************************************/
   CUDA_CHECK(cudaDeviceSynchronize());
   return 0;
}
