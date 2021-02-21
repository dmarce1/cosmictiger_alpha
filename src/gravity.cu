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

CUDA_DEVICE int cuda_cc_interactions(particle_set *parts, kick_params_type *params_ptr) {

   kick_params_type &params = *params_ptr;
   const int &tid = threadIdx.x;
   __shared__
   extern int shmem_ptr[];
   cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
   auto &Lreduce = shmem.Lreduce;
   auto &multis = params.multi_interactions;
   expansion<float> L;
   int flops = 0;
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
      multipole_interaction(L, mpole, fpos, false);
   }
   flops += multis.size() * FLOPS_CC;
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
   return flops;
}

CUDA_DEVICE int cuda_ewald_cc_interactions(particle_set *parts, kick_params_type *params_ptr,
      array<hifloat, KICK_BLOCK_SIZE> *lptr) {
   kick_params_type &params = *params_ptr;
   const int &tid = threadIdx.x;
   auto &Lreduce = *lptr;
   auto &multis = params.multi_interactions;
   expansion<hifloat> L;
   for (int i = 0; i < LP; i++) {
      L[i] = 0.0;
   }
   int flops = 0;
   const auto &pos = ((tree*) params.tptr)->pos;
   for (int i = tid; i < multis.size(); i += KICK_BLOCK_SIZE) {
      const multipole mpole_float = ((tree*) multis[i])->multi;
      multipole_type<hifloat> mpole;
      for (int i = 0; i < MP; i++) {
         mpole[i] = mpole_float[i];
      }
      array<hifloat, NDIM> fpos;
      for (int dim = 0; dim < NDIM; dim++) {
#ifdef EWALD_DOUBLE_PRECISION
         fpos[dim] = (fixed<int32_t>(pos[dim]) - fixed<int32_t>(pos[dim])).to_double();
#else
         fpos[dim] = (fixed<int32_t>(pos[dim]) - fixed<int32_t>(pos[dim])).to_float();
#endif
      }
      flops += multipole_interaction_ewald(L, mpole, fpos, false);
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
   return flops;
}

CUDA_DEVICE int cuda_cp_interactions(particle_set *parts, kick_params_type *params_ptr) {
   kick_params_type &params = *params_ptr;
   const int &tid = threadIdx.x;
   __shared__
   extern int shmem_ptr[];
   cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
   auto &Lreduce = shmem.Lreduce;
   auto &inters = params.part_interactions;
   const auto &sinks = ((tree*) params.tptr)->pos;
   auto &sources = shmem.src;
   const auto &myparts = ((tree*) params.tptr)->parts;
   size_t part_index;
   int flops = 0;
   expansion<float> L;
   if (inters.size() > 0) {
      for (int j = 0; j < LP; j++) {
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
            multipole_interaction(L, 1.0f, dx, false);
         }
         flops += (these_parts.second - these_parts.first) * FLOPS_CP;
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
   return flops;
}

CUDA_DEVICE int cuda_pp_interactions(particle_set *parts, kick_params_type *params_ptr) {
   kick_params_type &params = *params_ptr;
   const int &tid = threadIdx.x;
   __shared__
   extern int shmem_ptr[];
   cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
   auto &f = shmem.f;
   auto &F = params.F;
   auto &rungs = shmem.rungs;
   auto &sources = shmem.src;
   auto &sinks = shmem.sink;
   auto &inters = params.part_interactions;
   const auto h = params.hsoft;
   const auto h2 = h * h;
   const auto h2over4 = h2 / 4.0;
   const auto hinv = 1.0 / h;
   const auto h3inv = 1.0 / (h * h * h);
   int flops = 0;
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
                  float force;
                  const float rinv = rsqrtf(fmaxf(r2, h2over4)); // 8
                  if (r2 >= h2) {
                     force = rinv * rinv * rinv; // 2
                  } else {
                     const float r = 1.0f / rinv;
                     const float roh = r * hinv;                         // 2
                     const float roh2 = roh * roh;                           // 1
                     if (r2 > h2over4) {
                        const float roh3 = roh2 * roh;                         // 1
                        force = float(-32.0 / 3.0);
                        force = fmaf(force, roh, float(+192.0 / 5.0));                         // 2
                        force = fmaf(force, roh, float(-48.0));                         // 2
                        force = fmaf(force, roh, float(+64.0 / 3.0));                         // 2
                        force = fmaf(force, roh3, float(-1.0 / 15.0));                         // 2
                        force *= rinv * rinv * rinv;                         // 1
                     } else {
                        force = float(+32.0);
                        force = fmaf(force, roh, float(-192.0 / 5.0));                           // 2
                        force = fmaf(force, roh2, float(+32.0 / 3.0));                           // 2
                        force *= h3inv;                           // 1
                     }
                  }
                  for (int dim = 0; dim < NDIM; dim++) { // 3
                     f[dim][tid] = fmaf(dx[dim], force, f[dim][tid]);
                  }
               }
               flops += part_index * FLOPS_PP;
               __syncwarp();
               for (int P = KICK_BLOCK_SIZE / 2; P >= 1; P /= 2) {
                  if (tid < P) {
                     for (int dim = 0; dim < NDIM; dim++) {
                        f[dim][tid] += f[dim][tid + P];
                     }
                  }
                  __syncwarp();
               }
               if (tid == 0) {
                  for (int dim = 0; dim < NDIM; dim++) {
                     F[dim][k] -= f[dim][0];
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
   return flops;
}

CUDA_DEVICE
int cuda_pc_interactions(particle_set *parts, kick_params_type *params_ptr) {

   kick_params_type &params = *params_ptr;
   const int &tid = threadIdx.x;
   __shared__
   extern int shmem_ptr[];
   cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
   int flops = 0;
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
               multipole_interaction(Lforce, ((tree*) inters[i])->multi, dx, false);
               for (int dim = 0; dim < NDIM; dim++) {
                  f[dim][tid] -= Lforce[dim + 1];
               }
            }
            flops += inters.size() * FLOPS_PC;
            __syncwarp();
            for (int P = KICK_BLOCK_SIZE / 2; P >= 1; P /= 2) {
               if (tid < P) {
                  for (int dim = 0; dim < NDIM; dim++) {
                     f[dim][tid] += f[dim][tid + P];
                  }
               }
               __syncwarp();
            }
            if (tid == 0) {
               for (int dim = 0; dim < NDIM; dim++) {
                  F[dim][k] += f[dim][0];
               }
            }
         }
      }
   }
   return flops;
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
         array<hifloat, NDIM> dx;
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
