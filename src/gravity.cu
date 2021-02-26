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

CUDA_DEVICE int cuda_cc_interactions(particle_set *parts,const vector<tree_ptr>& multis,  kick_params_type *params_ptr) {

   kick_params_type &params = *params_ptr;
   const int &tid = threadIdx.x;
   __shared__
   volatile
   extern int shmem_ptr[];
   cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
   auto &Lreduce = shmem.Lreduce;
   if (multis.size() == 0) {
      return 0;
   }
   expansion<float> L;
   int flops = 0;
   for (int i = 0; i < LP; i++) {
      L[i] = 0.0;
   }
   const auto &pos = ((tree*) params.tptr)->pos;
   for (int i = tid; i < multis.size(); i += KICK_BLOCK_SIZE) {
      const multipole mpole = ((tree*) multis[i])->multi;
      array<float, NDIM> fpos;
#ifdef PERIODIC_OFF
      for (int dim = 0; dim < NDIM; dim++) {
         fpos[dim] = pos[dim].to_float() - ((tree*) multis[i])->pos[dim].to_float();
      }
#else
      for (int dim = 0; dim < NDIM; dim++) {
         fpos[dim] = (fixed<int32_t>(pos[dim]) - fixed<int32_t>(((tree*) multis[i])->pos[dim])).to_float();
      }
#endif
      multipole_interaction(L, mpole, fpos, false);
   }
   flops += multis.size() * FLOPS_CC;
   for (int i = 0; i < LP; i++) {
      Lreduce[tid] = L[i];
      __syncwarp();
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

#ifdef __CUDA_ARCH__

CUDA_DEVICE int cuda_ewald_cc_interactions(particle_set *parts, kick_params_type *params_ptr,
      array<hifloat, KICK_BLOCK_SIZE> *lptr) {
   kick_params_type &params = *params_ptr;
   const int &tid = threadIdx.x;
   auto &Lreduce = *lptr;
   auto &multis = params.multi_interactions;
   if( multis.size() == 0 ) {
      return 0;
   }
   expansion<hifloat> L;
   for (int i = 0; i < LP; i++) {
      L[i] = 0.0;
   }
   int flops = 0;
   const auto &pos = ((tree*) params.tptr)->pos;
   for (int i = tid; i < multis.size(); i += KICK_BLOCK_SIZE) {
      const multipole mpole_float = ((tree*) multis[i])->multi;
      multipole_type<hifloat> mpole;
      for (int j = 0; j < MP; j++) {
         mpole[j] = mpole_float[j];
      }
      array<hifloat, NDIM> fpos;
      for (int dim = 0; dim < NDIM; dim++) {
         fpos[dim] = (fixed<int32_t>(pos[dim]) - fixed<int32_t>(((tree*) multis[i])->pos[dim])).to_double();
      }
      flops += multipole_interaction_ewald(L, mpole, fpos, false);
   }
   for (int i = 0; i < LP; i++) {
      Lreduce[tid] = L[i];
      __syncwarp();
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

#endif

CUDA_DEVICE int cuda_cp_interactions(particle_set *parts, const vector<tree_ptr>& parti, kick_params_type *params_ptr) {
   kick_params_type &params = *params_ptr;
   const int &tid = threadIdx.x;
   __shared__
   volatile
   extern int shmem_ptr[];
   cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
   auto &Lreduce = shmem.Lreduce;
   if (parti.size() == 0) {
      return 0;
   }
   auto &sources = shmem.src;
   const auto &myparts = ((tree*) params.tptr)->parts;
   size_t part_index;
   int flops = 0;
   expansion<float> L;
   if (parti.size() > 0) {
      for (int j = 0; j < LP; j++) {
         L[j] = 0.0;
      }
      auto these_parts = ((tree*) parti[0])->parts;
      int i = 0;
      const auto &pos = ((tree*) params.tptr)->pos;
      while (i < parti.size()) {
         part_index = 0;
         while (part_index < KICK_PP_MAX && i < parti.size()) {
            while (i + 1 < parti.size()) {
               if (these_parts.second == ((tree*) parti[i + 1])->parts.first) {
                  these_parts.second = ((tree*) parti[i + 1])->parts.second;
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
               if (i < parti.size()) {
                  these_parts = ((tree*) parti[i])->parts;
               }
            }
         }
         for (int j = tid; j < part_index; j += KICK_BLOCK_SIZE) {
            array<float, NDIM> dx;
#ifdef PERIODIC_OFF
            for (int dim = 0; dim < NDIM; dim++) {
               dx[dim] = pos[dim].to_float() - sources[dim][j].to_float();
            }
#else
            for (int dim = 0; dim < NDIM; dim++) {
               dx[dim] = (fixed<int32_t>(pos[dim]) - fixed<int32_t>(sources[dim][j])).to_float();
            }
#endif
            multipole_interaction(L, 1.0f, dx, false);
         }
         flops += (these_parts.second - these_parts.first) * FLOPS_CP;
      }
      for (int i = 0; i < LP; i++) {
         Lreduce[tid] = L[i];
         __syncwarp();
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

CUDA_DEVICE int cuda_pp_interactions(particle_set *parts, const vector<tree_ptr>& parti, kick_params_type *params_ptr) {
   kick_params_type &params = *params_ptr;
   const int &tid = threadIdx.x;
   __shared__
   volatile
   extern int shmem_ptr[];
   cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
   auto &f = shmem.f;
   auto &F = params.F;
   auto &rungs = shmem.rungs;
   auto &sources = shmem.src;
   auto &sinks = shmem.sink;
   const auto h = params.hsoft;
   const auto h2 = h * h;
   const auto h2inv = 1.0 / h / h;
   int flops = 0;
   size_t part_index;
   if (parti.size() == 0) {
      return 0;
   }
//   printf( "%i\n", parti.size());
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
   auto these_parts = ((tree*) parti[0])->parts;
   while (i < parti.size()) {
      part_index = 0;
      while (part_index < KICK_PP_MAX && i < parti.size()) {
         while (i + 1 < parti.size()) {
            if (these_parts.second == ((tree*) parti[i + 1])->parts.first) {
               these_parts.second = ((tree*) parti[i + 1])->parts.second;
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
            if (i < parti.size()) {
               these_parts = ((tree*) parti[i])->parts;
            }
         }
      }
      __syncwarp();
      const auto offset = ((tree*) params.tptr)->parts.first;
      for (int k = 0; k < nsinks; k++) {
         if (rungs[k] >= params.rung || rungs[k] == -1) {
            for (int dim = 0; dim < NDIM; dim++) {
               f[dim][tid] = 0.f;
            }
            for (int j = tid; j < part_index; j += KICK_BLOCK_SIZE) {
//                 const auto tm = clock64();
               array<float, NDIM> dx;
#ifdef PERIODIC_OFF
               for (int dim = 0; dim < NDIM; dim++) { // 3
                  dx[dim] = sinks[dim][k].to_float() - sources[dim][j].to_float();
               }
#else
               for (int dim = 0; dim < NDIM; dim++) { // 3
                  dx[dim] = (fixed<int32_t>(sinks[dim][k]) - fixed<int32_t>(sources[dim][j])).to_float();
               }
#endif
               const auto r2 = fmaf(dx[0], dx[0], fmaf(dx[1], dx[1], sqr(dx[2]))); // 3
               float r3inv;
               if (r2 >= h2) {
                  const float rinv = rsqrt(r2); // 8
                  r3inv = rinv * rinv * rinv; // 2
               } else {
                  const float r2overh2 = r2 * h2inv;
                  r3inv = fmaf(r2overh2, (5.25f - 1.875f * r2overh2), -4.375f);
               }
               f[0][tid] = fmaf(dx[0], r3inv, f[0][tid]);
               f[1][tid] = fmaf(dx[1], r3inv, f[1][tid]);
               f[2][tid] = fmaf(dx[2], r3inv, f[2][tid]);
               //                 printf("%li \n", (clock64() - tm) / 2);
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
   }
   return flops;
}

CUDA_DEVICE
int cuda_pc_interactions(particle_set *parts, const vector<tree_ptr>& multis, kick_params_type *params_ptr) {
   kick_params_type &params = *params_ptr;
   const int &tid = threadIdx.x;
   __shared__
   volatile
   extern int shmem_ptr[];
   cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
   int flops = 0;
   auto &f = shmem.f;
   auto &F = params.F;
   auto &rungs = shmem.rungs;
   auto &sinks = shmem.sink;
   const auto &myparts = ((tree*) params.tptr)->parts;
   const int mmax = ((multis.size() - 1) / KICK_BLOCK_SIZE + 1) * KICK_BLOCK_SIZE;
   const int nparts = myparts.second - myparts.first;
   if (multis.size() == 0) {
      return 0;
   }
   for (int i = tid; i < nparts; i += KICK_BLOCK_SIZE) {
      rungs[i] = parts->rung(i + myparts.first);
      if (rungs[i] >= params.rung || rungs[i] == -1) {
         for (int dim = 0; dim < NDIM; dim++) {
            sinks[dim][i] = parts->pos(dim, myparts.first + i);
         }
      }
   }
   __syncwarp();
   for (int k = 0; k < nparts; k++) {
      if (rungs[k] >= params.rung || rungs[k] == -1) {
         for (int dim = 0; dim < NDIM; dim++) {
            f[dim][tid] = 0.f;
         }
         for (int i = tid; i < mmax; i += KICK_BLOCK_SIZE) {
            const auto &sources = ((tree*) multis[min(i, (int) multis.size() - 1)])->pos;
            if (i < multis.size()) {
               array<float, NDIM> dx;
               array<float, NDIM + 1> Lforce;
#ifdef PERIODIC_OFF
               for (int dim = 0; dim < NDIM; dim++) {
                  dx[dim] = sinks[dim][k].to_float() - sources[dim].to_float();
               }
#else
               for (int dim = 0; dim < NDIM; dim++) {
                  dx[dim] = (fixed<int32_t>(sinks[dim][k]) - fixed<int32_t>(sources[dim])).to_float();
               }
#endif
               multipole_interaction(Lforce, ((tree*) multis[i])->multi, dx, false);
               for (int dim = 0; dim < NDIM; dim++) {
                  f[dim][tid] += Lforce[dim + 1];
               }
            }
            flops += multis.size() * FLOPS_PC;
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
         if (tid == 0) {
            for (int dim = 0; dim < NDIM; dim++) {
               F[dim][k] -= f[dim][0];
            }
         }
      }
   }
   return flops;
}

#ifdef TEST_FORCE
//
//CUDA_DEVICE extern ewald_indices *four_indices_ptr;
//CUDA_DEVICE extern ewald_indices *real_indices_ptr;
//CUDA_DEVICE extern periodic_parts *periodic_parts_ptr;

CUDA_KERNEL cuda_pp_ewald_interactions(particle_set *parts, size_t *test_parts, float *err, float *norm) {
   const int &tid = threadIdx.x;
   const int &bid = blockIdx.x;
#ifndef PERIODIC_OFF
   const auto &hparts = *periodic_parts_ptr;
   const auto &four_indices = *four_indices_ptr;
   const auto &real_indices = *real_indices_ptr;
#endif

   const auto index = test_parts[bid];
   array<fixed32, NDIM> sink;
   for (int dim = 0; dim < NDIM; dim++) {
      sink[dim] = parts->pos(dim, index);
   }
   const auto f_x = parts->force(0, index);
   const auto f_y = parts->force(1, index);
   const auto f_z = parts->force(2, index);
   __shared__ array<array<float, KICK_BLOCK_SIZE>, NDIM>
   f;
   for (int dim = 0; dim < NDIM; dim++) {
      f[dim][tid] = 0.0;
   }
   for (size_t source = tid; source < parts->size(); source += KICK_BLOCK_SIZE) {
      if (source != index) {
         array<float, NDIM> X;
#ifdef PERIODIC_OFF
         for (int dim = 0; dim < NDIM; dim++) {
            X[dim] = sink[dim].to_float() - parts->pos(dim, source).to_float();
         }
         const float r2 = sqr(X[0]) + sqr(X[1]) + sqr(X[2]);
         const float rinv = rsqrt(r2);
         const float rinv3 = rinv * rinv * rinv;
         for (int dim = 0; dim < NDIM; dim++) {
            f[dim][tid] -= X[dim] * rinv3;
         }
#else
      for (int dim = 0; dim < NDIM; dim++) {
         X[dim] = (fixed<int32_t>(sink[dim]) - fixed<int32_t>(parts->pos(dim, source))).to_float();
      }
      for (int i = 0; i < real_indices.size(); i++) {
         const auto n = real_indices.get(i);
         array<hifloat, NDIM> dx;
         for (int dim = 0; dim < NDIM; dim++) {
            dx[dim] = X[dim] - n[dim];
         }
         const float r2 = sqr(dx[0]) + sqr(dx[1]) + sqr(dx[2]);
         if (r2 < (EWALD_REAL_CUTOFF * EWALD_REAL_CUTOFF)) {  // 1
            const float r = sqrt(r2);  // 1
            const float rinv = 1.f / r;  // 2
            const float r2inv = rinv * rinv;  // 1
            const float r3inv = r2inv * rinv;  // 1
            const float exp0 = expf(-4.f * r2);  // 26
            const float erfc0 = erfcf(2.f * r);                                    // 10
            const float expfactor = 4.0 / sqrt(M_PI) * r * exp0;  // 2
            const float d1 = (expfactor + erfc0) * r3inv;           // 2
            for (int dim = 0; dim < NDIM; dim++) {
               f[dim][tid] -= dx[dim] * d1;
            }
         }
      }
      for (int i = 0; i < four_indices.size(); i++) {
         const auto &h = four_indices.get(i);
         const auto &hpart = hparts.get(i);
         const float hdotx = h[0] * X[0] + h[1] * X[1] + h[2] * X[2];
         float so = sinf(2.0 * M_PI * hdotx);
         for (int dim = 0; dim < NDIM; dim++) {
            f[dim][tid] -= hpart(dim) * so;
         }
      }
#endif
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
   const auto f_ffm = sqrt(f_x * f_x + f_y * f_y + f_z * f_z);
   const auto f_dir = sqrt(sqr(f[0][0]) + sqr(f[1][0]) + sqr(f[2][0]));
   if (tid == 0) {
      norm[bid] = f_dir;
      err[bid] = fabsf(f_ffm - f_dir);
   }
}

void cuda_compare_with_direct(particle_set *parts) {
   size_t *test_parts;
   float *errs;
   float *norms;
   CUDA_MALLOC(test_parts, N_TEST_PARTS);
   CUDA_MALLOC(errs, N_TEST_PARTS);
   CUDA_MALLOC(norms, N_TEST_PARTS);
   const size_t nparts = parts->size();
   for (int i = 0; i < N_TEST_PARTS; i++) {
      test_parts[i] = rand() % nparts;
   }
cuda_pp_ewald_interactions<<<N_TEST_PARTS,KICK_BLOCK_SIZE>>>(parts, test_parts, errs, norms);
                                 CUDA_CHECK(cudaDeviceSynchronize());
   float avg_err = 0.0;
   float norm = 0.0;
   for (int i = 0; i < N_TEST_PARTS; i++) {
      avg_err += errs[i];
      norm += norms[i];
   }
   avg_err /= norm;
//   avg_err /= norm;
   printf("Error is %e\n", avg_err);
   CUDA_FREE(norms);
   CUDA_FREE(errs);
   CUDA_FREE(test_parts);
}

#endif
