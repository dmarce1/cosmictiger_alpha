/*
 * gravity.cu
 *
 *  Created on: Feb 10, 2021
 *      Author: dmarce1
 */

#include <cosmictiger/gravity.hpp>
#include <cosmictiger/array.hpp>

__managed__ double pp_crit1_time;
__managed__ double pp_crit2_time;

CUDA_DEVICE void cuda_cc_interactions(particle_set *parts, kick_params_type *params_ptr) {

   kick_params_type &params = *params_ptr;
   const int &tid = threadIdx.x;
   __shared__
   extern int shmem_ptr[];
   cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
   auto &flops = shmem.flops;
   auto &Lreduce = shmem.Lreduce;
   auto &multis = params.multi_interactions;
   expansion<accum_real> L;
   for (int i = 0; i < LP; i++) {
      L[i] = 0.0;
   }
   const auto &pos = ((tree*) params.tptr)->pos;
   for (int i = tid; i < multis.size(); i += KICK_BLOCK_SIZE) {
      const multipole mpole = ((tree*) multis[i])->multi;
      array<float, NDIM> fpos;
      for (int dim = 0; dim < NDIM; dim++) {
         fpos[dim] = (fixed<int32_t>(pos[dim]) - fixed<int32_t>(pos[dim])).to_float();
      }
      flops[tid] += NDIM;
      flops[tid] += multipole_interaction(L, mpole, fpos, false);
   }
   __syncwarp();
   for (int i = 0; i < LP; i++) {
      Lreduce[tid] = L[i];
      for (int P = KICK_BLOCK_SIZE / 2; P >= 1; P /= 2) {
         if (tid < P) {
            Lreduce[tid] += Lreduce[tid + P];
         }
         __syncwarp();
      }
      if (tid == 0) {
         params.L[params.depth][i] += Lreduce[0];
      }
   }
}

CUDA_DEVICE int cuda_ewald_cc_interactions(particle_set *parts, kick_params_type *params_ptr,
      array<accum_real, KICK_BLOCK_SIZE> *lptr, array<int32_t, KICK_BLOCK_SIZE> *fptr) {
   kick_params_type &params = *params_ptr;
   const int &tid = threadIdx.x;
   auto &flops = *fptr;
   auto &Lreduce = *lptr;
   auto &multis = params.multi_interactions;
   expansion<accum_real> L;
   for (int i = 0; i < LP; i++) {
      L[i] = 0.0;
   }
   flops[tid] = 0;
   __syncwarp();
   const auto &pos = ((tree*) params.tptr)->pos;
   for (int i = tid; i < multis.size(); i += KICK_BLOCK_SIZE) {
      const multipole mpole_float = ((tree*) multis[i])->multi;
      multipole_type<ewald_real> mpole;
      for (int i = 0; i < MP; i++) {
         mpole[i] = mpole_float[i];
      }
      array<ewald_real, NDIM> fpos;
      for (int dim = 0; dim < NDIM; dim++) {
#ifdef EWALD_DOUBLE_PRECISION
         fpos[dim] = (fixed<int32_t>(pos[dim]) - fixed<int32_t>(pos[dim])).to_double();
#else
         fpos[dim] = (fixed<int32_t>(pos[dim]) - fixed<int32_t>(pos[dim])).to_float();
#endif
      }
      flops[tid] += 3;
      flops[tid] += multipole_interaction_ewald(L, mpole, fpos, false);
   }
   __syncwarp();
   for (int i = 0; i < LP; i++) {
      Lreduce[tid] = L[i];
      for (int P = KICK_BLOCK_SIZE / 2; P >= 1; P /= 2) {
         if (tid < P) {
            Lreduce[tid] += Lreduce[tid + P];
         }
         __syncwarp();
      }
      if (tid == 0) {
         params.L[params.depth][i] += Lreduce[0];
      }
   }
   return flops[0];
}

CUDA_DEVICE void cuda_cp_interactions(particle_set *parts, kick_params_type *params_ptr) {
   kick_params_type &params = *params_ptr;
   const int &tid = threadIdx.x;
   __shared__
   extern int shmem_ptr[];
   cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
   auto &flops = shmem.flops;
   auto &Lreduce = shmem.Lreduce;
   auto &inters = params.part_interactions;
   const auto &sinks = ((tree*) params.tptr)->pos;
   auto &sources = shmem.src;
   const auto &myparts = ((tree*) params.tptr)->parts;
   size_t part_index;
   expansion<accum_real> L;
   if (inters.size() > 0) {
      for( int j = 0; j < LP; j++) {
         L[j] = 0.0;
      }
      auto these_parts = ((tree*) inters[0])->parts;
      int i = 0;
      while (i < inters.size()) {
         part_index = 0;
         while (part_index < KICK_PP_MAX && i < inters.size()) {
            while (i + 1 < inters.size()) {
               if (these_parts.second == ((tree*) inters[i + 1])->parts.first) {
                  these_parts.second = ((tree*) inters[i + 1])->parts.second;
                  i++;
               } else {
                  break;
               }
            }
            const size_t imin = these_parts.first;
            const size_t imax = min(these_parts.first + (KICK_PP_MAX - part_index), these_parts.second);
            for (size_t j = imin + tid; j < imax; j += KICK_BLOCK_SIZE) {
               for (int dim = 0; dim < NDIM; dim++) {
                  sources[dim][part_index + j - imin] = parts->pos(dim, j);
               }
            }
            these_parts.first += imax - imin;
            part_index += imax - imin;
            if (these_parts.first == these_parts.second) {
               i++;
               if (i < inters.size()) {
                  these_parts = ((tree*) inters[i])->parts;
               }
            }
         }
         for (int j = these_parts.first + tid; j < these_parts.second; j += KICK_BLOCK_SIZE) {
            array<float, NDIM> dx;
            for (int dim = 0; dim < NDIM; dim++) {
               dx[dim] = (fixed<int32_t>(parts->pos(dim, j)) - fixed<int32_t>(sinks[dim])).to_float();
            }
            flops[tid] += NDIM;
            expansion<float> L;
            flops[tid] += multipole_interaction(L, 1.0f, dx, false);
            flops[tid] += LP;
         }
      }
      __syncwarp();
      for (int i = 0; i < LP; i++) {
         Lreduce[tid] = L[i];
         for (int P = KICK_BLOCK_SIZE / 2; P >= 1; P /= 2) {
            if (tid < P) {
               Lreduce[tid] += Lreduce[tid + P];
            }
            __syncwarp();
         }
         if (tid == 0) {
            params.L[params.depth][i] += Lreduce[0];
         }
      }
   }
}

CUDA_DEVICE void cuda_pp_interactions(particle_set *parts, kick_params_type *params_ptr) {
   kick_params_type &params = *params_ptr;
   const int &tid = threadIdx.x;
   __shared__
   extern int shmem_ptr[];
   cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
   auto &f = shmem.f;
   auto &F = params.F;
   auto &rungs = shmem.rungs;
#ifdef COUNT_FLOPS
   auto &flops = shmem.flops;
#endif
   auto &sources = shmem.src;
   auto &sinks = shmem.sink;
   auto &inters = params.part_interactions;
   const auto h2 = sqr(params.hsoft);
   size_t part_index;
   if (inters.size()) {
      const auto &myparts = ((tree*) params.tptr)->parts;
      const size_t nsinks = myparts.second - myparts.first;
      for (int i = tid; i < nsinks; i += KICK_BLOCK_SIZE) {
         rungs[i] = parts->rung(i + myparts.first);
         if (rungs[i] >= params.rung || rungs[i] == -1) {
            for (int dim = 0; dim < NDIM; dim++) {
               sinks[dim][i] = parts->pos(dim, i + myparts.first);
            }
         }
      }
      int i = 0;
      __syncwarp();
      auto these_parts = ((tree*) inters[0])->parts;
      while (i < inters.size()) {
#ifdef TIMINGS
         uint64_t tm = clock64();
#endif
         part_index = 0;
         while (part_index < KICK_PP_MAX && i < inters.size()) {
            while (i + 1 < inters.size()) {
               if (these_parts.second == ((tree*) inters[i + 1])->parts.first) {
                  these_parts.second = ((tree*) inters[i + 1])->parts.second;
                  i++;
               } else {
                  break;
               }
            }
            const size_t imin = these_parts.first;
            const size_t imax = min(these_parts.first + (KICK_PP_MAX - part_index), these_parts.second);
            for (size_t j = imin + tid; j < imax; j += KICK_BLOCK_SIZE) {
               for (int dim = 0; dim < NDIM; dim++) {
                  sources[dim][part_index + j - imin] = parts->pos(dim, j);
               }
            }
            these_parts.first += imax - imin;
            part_index += imax - imin;
            if (these_parts.first == these_parts.second) {
               i++;
               if (i < inters.size()) {
                  these_parts = ((tree*) inters[i])->parts;
               }
            }
         }
#ifdef TIMINGS
         if( tid == 0 ) {
            atomicAdd(&pp_crit1_time, (double)(clock64()-tm));
         }
         tm = clock64();
#endif
         __syncwarp();
         const auto offset = ((tree*) params.tptr)->parts.first;
         for (int k = 0; k < nsinks; k++) {
            if (rungs[k] >= params.rung || rungs[k] == -1) {
               for (int dim = 0; dim < NDIM; dim++) {
                  f[dim][tid] = 0.f;
               }
               for (int j = tid; j < part_index; j += KICK_BLOCK_SIZE) {
                  array<float, NDIM> dx;
                  for (int dim = 0; dim < NDIM; dim++) { // 3
                     dx[dim] = (fixed<int32_t>(sources[dim][j]) - fixed<int32_t>(sinks[dim][k])).to_float();
                  }
                  const auto r2 = fmaf(dx[0], dx[0], fmaf(dx[1], dx[1], sqr(dx[2]))); // 3
                  const auto rinv = rsqrtf(fmaxf(r2, h2)); // 8
                  const auto rinv3 = rinv * rinv * rinv; // 2
                  for (int dim = 0; dim < NDIM; dim++) { // 3
                     f[dim][tid] = fmaf(dx[dim], rinv3, f[dim][tid]);
                  }
                  flops[tid] += 46;
               }
               __syncwarp();
               for (int P = KICK_BLOCK_SIZE / 2; P >= 1; P /= 2) {
                  if (tid < P) {
                     for (int dim = 0; dim < NDIM; dim++) {
                        f[dim][tid] += f[dim][tid + P];
                        flops[tid]++;
                     }
                  }
                  __syncwarp();
               }
               if (tid == 0) {
                  for (int dim = 0; dim < NDIM; dim++) {
                     F[dim][k] -= f[dim][0];
#ifdef COUNT_FLOPS
                     flops[tid]++;
#endif
                  }
               }
            }
         }
#ifdef TIMINGS
         if( tid == 0 ) {
            atomicAdd(&pp_crit2_time, (double)(clock64()-tm));
         }
#endif
      }
   }
}

CUDA_DEVICE
void cuda_pc_interactions(particle_set *parts, kick_params_type *params_ptr) {

   kick_params_type &params = *params_ptr;
   const int &tid = threadIdx.x;
   __shared__
   extern int shmem_ptr[];
   cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
   auto &flops = shmem.flops;
   auto &f = shmem.f;
   auto &F = params.F;
   auto &rungs = shmem.rungs;
   auto &sinks = shmem.sink;
   auto &inters = params.multi_interactions;
   const auto &myparts = ((tree*) params.tptr)->parts;
   const int mmax = ((inters.size() - 1) / KICK_BLOCK_SIZE + 1) * KICK_BLOCK_SIZE;
   const int nparts = myparts.second - myparts.first;
   for (int i = tid; i < nparts; i += KICK_BLOCK_SIZE) {
      rungs[i] = parts->rung(i + myparts.first);
      if (rungs[i] >= params.rung || rungs[i] == -1) {
         for (int dim = 0; dim < NDIM; dim++) {
            sinks[dim][i] = parts->pos(dim, myparts.first + i);
         }
      }
   }
   for (int i = tid; i < mmax; i += KICK_BLOCK_SIZE) {
      const auto &sources = ((tree*) inters[min(i, (int) inters.size() - 1)])->pos;
      const int nparts = myparts.second - myparts.first;
      for (int k = 0; k < nparts; k++) {
         if (rungs[k] >= params.rung || rungs[k] == -1) {
            for (int dim = 0; dim < NDIM; dim++) {
               f[dim][tid] = 0.f;
            }
            if (i < inters.size()) {
               array<float, NDIM> dx;
               array<float, NDIM + 1> Lforce;
               for (int l = 0; l < NDIM + 1; l++) {
                  Lforce[l] = 0.f;
               }
               for (int dim = 0; dim < NDIM; dim++) {
                  dx[dim] = (fixed<int32_t>(sources[dim]) - fixed<int32_t>(sinks[dim][k])).to_float();
               }
               flops[tid] += NDIM;
               flops[tid] += multipole_interaction(Lforce, ((tree*) inters[i])->multi, dx, false);
               for (int dim = 0; dim < NDIM; dim++) {
                  f[dim][tid] -= Lforce[dim + 1];
               }
               flops[tid] += NDIM;
            }
            __syncwarp();
            for (int P = KICK_BLOCK_SIZE / 2; P >= 1; P /= 2) {
               if (tid < P) {
                  for (int dim = 0; dim < NDIM; dim++) {
                     f[dim][tid] += f[dim][tid + P];
                  }
                  flops[tid] += NDIM;
               }
               __syncwarp();
            }
            if (tid == 0) {
               for (int dim = 0; dim < NDIM; dim++) {
                  F[dim][k] += f[dim][0];
               }
               flops[tid] += NDIM;
            }
         }
      }
   }
}

#ifdef TEST_FORCE

CUDA_DEVICE extern ewald_indices *four_indices_ptr;
CUDA_DEVICE extern ewald_indices *real_indices_ptr;
CUDA_DEVICE extern periodic_parts *periodic_parts_ptr;

CUDA_KERNEL cuda_pp_ewald_interactions(particle_set *parts, size_t *test_parts, array<float, NDIM> *res) {
   const int &tid = threadIdx.x;
   const int &bid = blockIdx.x;
   const auto &hparts = *periodic_parts_ptr;
   const auto &four_indices = *four_indices_ptr;
   const auto &real_indices = *real_indices_ptr;

   const auto index = test_parts[bid];
   const auto src_x = parts->pos(0, index).to_float();
   const auto src_y = parts->pos(1, index).to_float();
   const auto src_z = parts->pos(2, index).to_float();
   __shared__ array<array<float, NDIM>, KICK_BLOCK_SIZE>
   f;
   for (int dim = 0; dim < NDIM; dim++) {
      f[dim][tid] = 0.0;
   }
   for (size_t sink = tid; sink < parts->size(); sink += KICK_BLOCK_SIZE) {
      if (sink == index) {
         continue;
      }
      array<float, NDIM> X;
      X[0] = src_x;
      X[1] = src_y;
      X[2] = src_z;
      for (int i = 0; i < real_indices.size(); i++) {
         const auto n = real_indices.get(i);
         array<ewald_real, NDIM> dx;
         for( int dim = 0; dim < NDIM; dim++) {
            dx[dim] = X[dim] - n[dim];
         }
         const float r2 = sqr(dx[0]) + sqr(dx[1]) + sqr(dx[2]);
         if (r2 < (EWALD_REAL_CUTOFF * EWALD_REAL_CUTOFF)) {  // 1
            const float r = sqrt(r2);  // 1
            const float cmask = 1.f - ((sqr(n[0])+sqr(n[1])+sqr(n[2])) > 0.0);  // 7
            const float rinv = 1.f / r;  // 2
            const float r2inv = rinv * rinv;  // 1
            const float r3inv = r2inv * rinv;  // 1
            const float exp0 = expf(-4.f * r2);  // 26
            const float erfc0 = erfcf(2.f * r);                                    // 10
            const float expfactor = 4.0 / sqrtf(M_PI) * r * exp0;  // 2
            const float e1 = expfactor * r3inv;  // 1
            const float d0 = -erfc0 * rinv;  // 2
            const float d1 = fma(-d0, r2inv, e1);  // 3
            for (int dim = 0; dim < NDIM; dim++) {
               f[dim][tid] = dx[dim] * d1;
            }
         }
      }
      for (int i = 0; i < four_indices.size(); i++) {
         const auto &h = four_indices.get(i);
         const auto &hpart = hparts.get(i);
         const float h2 = sqrt(h[0]) + sqr(h[1]) + sqr(h[2]);
         const float hdotx = h[0] * X[0] + h[1] * X[1] + h[2] * X[2];
         float so = sinf(2.0 * M_PI * hdotx);
         for (int dim = 0; dim < NDIM; dim++) {
            f[dim][tid] -= hpart(dim) * so;
         }
      }
   }
   __syncwarp();
   for (int P = KICK_BLOCK_SIZE / 2; P >= 1; P /= 2) {
      if (tid < P) {
         for (int dim = 0; dim < NDIM; dim++) {
            f[dim][tid] += f[dim][tid + P];
         }
      }
      __syncwarp();
   }
   for (int dim = 0; dim < NDIM; dim++) {
      res[index][dim] = f[dim][0];
   }
}

#endif
