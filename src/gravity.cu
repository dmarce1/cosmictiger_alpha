/*
 * gravity.cu
 *
 *  Created on: Feb 10, 2021
 *      Author: dmarce1
 */




#include <cosmictiger/gravity.hpp>



CUDA_DEVICE void cuda_cc_interactions(particle_set* parts, kick_params_type *params_ptr ) {
   kick_params_type &params = *params_ptr;
   const int &tid = threadIdx.x;
   __shared__
   extern int shmem_ptr[];
   cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
   auto &flops = shmem.flops;
   auto &Lreduce = shmem.Lreduce;
   auto &multis = params.multi_interactions;
   for (int i = 0; i < LP; i++) {
      Lreduce[tid][i] = 0.0;
   }
   __syncthreads();
   const auto &pos = ((tree*) params.tptr)->pos;
   for (int i = tid; i < params.nmulti; i += KICK_BLOCK_SIZE) {
      const multipole mpole = *((tree*) multis[i])->multi;
      expansion<float> L;
      array<float, NDIM> fpos;
      for (int dim = 0; dim < NDIM; dim++) {
         fpos[dim] = (fixed<int32_t>(pos[dim]) - fixed<int32_t>(pos[dim])).to_float();
      }
      flops[tid] += NDIM;
      flops[tid] += multipole_interaction(L, mpole, fpos, false);
      for (int j = 0; j < LP; j++) {
         Lreduce[tid][j] += L[j];
      }
      flops[tid] += LP;
   }
   __syncthreads();
   for (int P = KICK_BLOCK_SIZE / 2; P >= 1; P /= 2) {
      if (tid < P) {
         for (int i = 0; i < LP; i++) {
            Lreduce[tid][i] += Lreduce[tid + P][i];
         }
         flops[tid] += LP;
      }
      __syncthreads();
   }
   for (int i = tid; i < LP; i += KICK_BLOCK_SIZE) {
      params.L[params.depth][i] += Lreduce[0][i];
      flops[tid]++;
   }
}

CUDA_DEVICE int cuda_ewald_cc_interactions(particle_set* parts, kick_params_type *params_ptr ) {
   kick_params_type &params = *params_ptr;
   const int &tid = threadIdx.x;
   __shared__
   extern int shmem_ptr[];
   cuda_ewald_shmem &shmem = *(cuda_ewald_shmem*) shmem_ptr;
   auto &flops = shmem.flops;
   auto &Lreduce = shmem.Lreduce;
   auto &multis = params.multi_interactions;
   for (int i = 0; i < LP; i++) {
      Lreduce[tid][i] = 0.0;
   }
   flops[tid] = 0;
   __syncthreads();
   const auto &pos = ((tree*) params.tptr)->pos;
   for (int i = tid; i < params.nmulti; i += KICK_BLOCK_SIZE) {
      const multipole mpole_float = *((tree*) multis[i])->multi;
      multipole_type<ewald_real> mpole;
      for (int i = 0; i < MP; i++) {
         mpole[i] = mpole_float[i];
      }
      expansion<ewald_real> L;
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
      for (int j = 0; j < LP; j++) {
         Lreduce[tid][j] += L[j];
      }
      flops[tid] += 17;
   }
   __syncthreads();
   for (int P = KICK_BLOCK_SIZE / 2; P >= 1; P /= 2) {
      if (tid < P) {
         for (int i = 0; i < LP; i++) {
            Lreduce[tid][i] += Lreduce[tid + P][i];
            flops[tid]++;
         }
      }
      __syncthreads();
   }
   for (int i = tid; i < LP; i += KICK_BLOCK_SIZE) {
      params.L[params.depth][i] += Lreduce[0][i];
      flops[0]++;
   }
   return flops[0];
}

CUDA_DEVICE void cuda_cp_interactions(particle_set* parts, kick_params_type *params_ptr) {
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
   int i = 0;
   __syncthreads();
   if (params.npart > 0) {
      auto these_parts = ((tree*) inters[0])->parts;
      while (i < params.npart) {
         part_index = 0;
         while (part_index < KICK_PP_MAX && i < params.npart) {
            while (i + 1 < params.npart) {
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
               if (i < params.npart) {
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
            for (int j = 0; j < LP; j++) {
               Lreduce[tid][j] += L[j];
            }
            flops[tid] += LP;
         }
         __syncthreads();
         for (int P = KICK_BLOCK_SIZE / 2; P >= 1; P /= 2) {
            if (tid < P) {
               for (int i = 0; i < LP; i++) {
                  Lreduce[tid][i] += Lreduce[tid + P][i];
               }
               flops[tid] += LP;
            }
            __syncthreads();
         }
         for (int i = tid; i < LP; i += KICK_BLOCK_SIZE) {
            params.L[params.depth][i] += Lreduce[0][i];
            flops[tid]++;
         }
      }
   }
}

CUDA_DEVICE void cuda_pp_interactions(particle_set* parts, kick_params_type *params_ptr) {
   kick_params_type &params = *params_ptr;
   const int &tid = threadIdx.x;
   __shared__
   extern int shmem_ptr[];
   cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
   auto &f = shmem.f;
   auto &F = shmem.F;
#ifdef COUNT_FLOPS
   auto &flops = shmem.flops;
#endif
   auto &sources = shmem.src;
   auto &sinks = shmem.sink;
   auto &inters = params.part_interactions;
   const auto h2 = sqr(params.hsoft);
   size_t part_index;
   if (params.npart) {
      const auto &myparts = ((tree*) params.tptr)->parts;
      const size_t nsinks = myparts.second - myparts.first;
      for (int i = tid; i < nsinks; i += KICK_BLOCK_SIZE) {
         const auto this_rung = parts->rung(i + myparts.first);
         if (this_rung >= params.rung || this_rung == -1) {
            for (int dim = 0; dim < NDIM; dim++) {
               sinks[dim][i] = parts->pos(dim, i + myparts.first);
            }
         }
      }
      int i = 0;
      __syncthreads();
      auto these_parts = ((tree*) inters[0])->parts;
      while (i < params.npart) {
         part_index = 0;
         while (part_index < KICK_PP_MAX && i < params.npart) {
            while (i + 1 < params.npart) {
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
               if (i < params.npart) {
                  these_parts = ((tree*) inters[i])->parts;
               }
            }
         }
         __syncthreads();
         const auto offset = ((tree*) params.tptr)->parts.first;
         for (int k = 0; k < nsinks; k++) {
            const auto this_rung = parts->rung(k + offset);
            if (this_rung >= params.rung || this_rung == -1) {
               for (int dim = 0; dim < NDIM; dim++) {
                  f[dim][tid] = 0.f;
               }
               for (int j = tid; j < part_index; j += KICK_BLOCK_SIZE) {
                  array<float, NDIM> dx;
                  for (int dim = 0; dim < NDIM; dim++) { // 3
                     dx[dim] = (fixed<int32_t>(sources[dim][j]) - fixed<int32_t>(sinks[dim][k])).to_float();
                  }
                  const auto r2 = sqr(dx[0]) + sqr(dx[1]) + sqr(dx[2]); // 5
                  const auto rinv = rsqrtf(fmaxf(r2, h2)); // 8
                  const auto rinv3 = rinv * rinv * rinv; // 2
                  for (int dim = 0; dim < NDIM; dim++) { // 6
                     f[dim][tid] -= dx[dim] * rinv3;
                  }
#ifdef COUNT_FLOPS
                  flops[tid] += 24;
#endif
               }
               __syncthreads();
               for (int P = KICK_BLOCK_SIZE / 2; P >= 1; P /= 2) {
                  if (tid < P) {
                     for (int dim = 0; dim < NDIM; dim++) {
                        f[dim][tid] += f[dim][tid + P];
#ifdef COUNT_FLOPS
                        flops[tid]++;
#endif
                     }
                  }
                  __syncthreads();
               }
               if (tid == 0) {
                  for (int dim = 0; dim < NDIM; dim++) {
                     F[dim][k] += f[dim][0];
#ifdef COUNT_FLOPS
                     flops[tid]++;
#endif
                  }
               }
               __syncthreads();
            }
         }
      }
   }
}

CUDA_DEVICE
void cuda_pc_interactions(particle_set* parts, kick_params_type *params_ptr) {
   kick_params_type &params = *params_ptr;
   const int &tid = threadIdx.x;
   __shared__
   extern int shmem_ptr[];
   cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
   auto &flops = shmem.flops;
   auto &f = shmem.f;
   auto &F = shmem.F;
   auto &sinks = shmem.sink;
   auto &inters = params.multi_interactions;
   const auto &myparts = ((tree*) params.tptr)->parts;
   const auto offset = myparts.first;
   const int mmax = ((params.nmulti - 1) / KICK_BLOCK_SIZE + 1) * KICK_BLOCK_SIZE;
   const int nparts = myparts.second - myparts.first;
   for (int i = tid; i < nparts; i += KICK_BLOCK_SIZE) {
      const auto this_rung = parts->rung(i + myparts.first);
      if (this_rung >= params.rung || this_rung == -1) {
         for (int dim = 0; dim < NDIM; dim++) {
            sinks[dim][i] = parts->pos(dim, myparts.first + i);
         }
      }
   }
   for (int i = tid; i < mmax; i += KICK_BLOCK_SIZE) {
      const auto &sources = ((tree*) inters[i])->pos;
      const int nparts = myparts.second - myparts.first;
      for (int k = 0; k < nparts; k++) {
         const auto this_rung = parts->rung(k + offset);
         if (this_rung >= params.rung || this_rung == -1) {
            for (int dim = 0; dim < NDIM; dim++) {
               f[dim][tid] = 0.f;
            }
            __syncthreads();
            if (i < params.nmulti) {
               array<float, NDIM> dx;
               array<float, NDIM + 1> Lforce;
               for (int dim = 0; dim < NDIM; dim++) {
                  dx[dim] = (fixed<int32_t>(sources[dim]) - fixed<int32_t>(sinks[dim][k])).to_float();
               }
               flops[tid] += NDIM;
               flops[tid] += multipole_interaction(Lforce, *((tree*) inters[i])->multi, dx, false);
               for (int dim = 0; dim < NDIM; dim++) {
                  f[dim][tid] -= Lforce[dim + 1];
               }
               flops[tid] += NDIM;
            }
            __syncthreads();
            for (int P = KICK_BLOCK_SIZE / 2; P >= 1; P /= 2) {
               if (tid < P) {
                  for (int dim = 0; dim < NDIM; dim++) {
                     f[dim][tid] += f[dim][tid + P];
                  }
                  flops[tid] += NDIM;
               }
               __syncthreads();
            }
            if (tid == 0) {
               for (int dim = 0; dim < NDIM; dim++) {
                  F[dim][k] += f[dim][0];
               }
               flops[tid] += NDIM;
            }
            __syncthreads();
         }
      }
   }
}
