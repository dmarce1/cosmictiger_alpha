struct ewald_indices;
struct periodic_parts;
#include <cosmictiger/cuda.hpp>
#include <stack>

CUDA_DEVICE ewald_indices *four_indices_ptr;
CUDA_DEVICE ewald_indices *real_indices_ptr;
CUDA_DEVICE periodic_parts *periodic_parts_ptr;

#define TREECU
#include <cosmictiger/tree.hpp>
#include <cosmictiger/interactions.hpp>
#include <functional>

//CUDA_KERNEL cuda_kick()

#define CC_CP_DIRECT 0
#define CC_CP_EWALD 1
#define PC_PP_DIRECT 2
#define PC_PP_EWALD 3
#define N_INTERACTION_TYPES 4

CUDA_DEVICE particle_set *parts;

#define NITERS 4
#define MI 0
#define CI 1
#define OI 2
#define PI 3

using indices_array = array<array<int8_t, KICK_BLOCK_SIZE + 1>, NITERS>;
using counts_array = array<int16_t, NITERS>;

using pos_array = array<fixed32,KICK_PP_MAX>;

struct cuda_kick_shmem {
   indices_array indices;
   counts_array count;
   array<array<accum_real, KICK_BLOCK_SIZE>, NDIM> f;
   array<array<accum_real, MAX_BUCKET_SIZE>, NDIM> F;
   array<array<fixed32, KICK_PP_MAX>, NDIM> src;
   array<array<fixed32, MAX_BUCKET_SIZE>, NDIM> sink;
   array<expansion<accum_real>, KICK_BLOCK_SIZE> Lreduce;
};

struct cuda_ewald_shmem {
   array<expansion<accum_real>, KICK_BLOCK_SIZE> Lreduce;
};

CUDA_DEVICE void cuda_cc_interactions(kick_params_type *params_ptr, bool ewald = false) {
   kick_params_type &params = *params_ptr;
   const int &tid = threadIdx.x;
   __shared__
   extern int shmem_ptr[];
   cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
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
      multipole_interaction(L, mpole, fpos, ewald, false);
      for (int j = 0; j < LP; j++) {
         Lreduce[tid][j] += L[j];
      }
   }
   __syncthreads();
   for (int P = KICK_BLOCK_SIZE / 2; P >= 1; P /= 2) {
      if (tid < P) {
         for (int i = 0; i < LP; i++) {
            Lreduce[tid][i] += Lreduce[tid + P][i];
         }
      }
      __syncthreads();
   }
   for (int i = tid; i < LP; i += KICK_BLOCK_SIZE) {
      params.L[params.depth][i] += Lreduce[0][i];
   }
}

CUDA_DEVICE void cuda_ewald_cc_interactions(kick_params_type *params_ptr, bool ewald = false) {
   kick_params_type &params = *params_ptr;
   const int &tid = threadIdx.x;
   __shared__
   extern int shmem_ptr[];
   cuda_ewald_shmem &shmem = *(cuda_ewald_shmem*) shmem_ptr;
   auto &Lreduce = shmem.Lreduce;
   auto &multis = params.multi_interactions;
   for (int i = 0; i < LP; i++) {
      Lreduce[tid][i] = 0.0;
   }
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
         fpos[dim] = (fixed<int32_t>(pos[dim]) - fixed<int32_t>(pos[dim])).to_double();
      }
      multipole_interaction_ewald(L, mpole, fpos, ewald, false);
      for (int j = 0; j < LP; j++) {
         Lreduce[tid][j] += L[j];
      }
   }
   __syncthreads();
   for (int P = KICK_BLOCK_SIZE / 2; P >= 1; P /= 2) {
      if (tid < P) {
         for (int i = 0; i < LP; i++) {
            Lreduce[tid][i] += Lreduce[tid + P][i];
         }
      }
      __syncthreads();
   }
   for (int i = tid; i < LP; i += KICK_BLOCK_SIZE) {
      params.L[params.depth][i] += Lreduce[0][i];
   }
}

CUDA_DEVICE void cuda_cp_interactions(kick_params_type *params_ptr) {
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
            expansion<float> L;
            multipole_interaction(L, 1.0f, dx, false);
            for (int j = 0; j < LP; j++) {
               Lreduce[tid][j] += L[j];
            }
         }
         __syncthreads();
         for (int P = KICK_BLOCK_SIZE / 2; P >= 1; P /= 2) {
            if (tid < P) {
               for (int i = 0; i < LP; i++) {
                  Lreduce[tid][i] += Lreduce[tid + P][i];
               }
            }
            __syncthreads();
         }
         for (int i = tid; i < LP; i += KICK_BLOCK_SIZE) {
            params.L[params.depth][i] += Lreduce[0][i];
         }
      }
   }
}

CUDA_DEVICE void cuda_pp_interactions(kick_params_type *params_ptr) {
   kick_params_type &params = *params_ptr;
   const int &tid = threadIdx.x;
   __shared__
   extern int shmem_ptr[];
   cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
   auto &f = shmem.f;
   auto &F = shmem.F;
   auto &sources = shmem.src;
   auto &sinks = shmem.sink;
   auto &inters = params.part_interactions;
   size_t part_index;
   if (params.npart) {
      const auto &myparts = ((tree*) params.tptr)->parts;
      const size_t nsinks = myparts.second - myparts.first;
      for (int i = tid; i < nsinks; i += KICK_BLOCK_SIZE) {
         for (int dim = 0; dim < NDIM; dim++) {
            sinks[dim][i] = parts->pos(dim, i + myparts.first);
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
            if (parts->rung(k + offset) >= 0) {
               for (int dim = 0; dim < NDIM; dim++) {
                  f[dim][tid] = 0.f;
               }
               for (int j = tid; j < part_index; j += KICK_BLOCK_SIZE) {
                  array<float, NDIM> dx;
                  for (int dim = 0; dim < NDIM; dim++) {
                     dx[dim] = (fixed<int32_t>(sources[dim][j]) - fixed<int32_t>(sinks[dim][k])).to_float();
                  }
                  const auto r2 = sqr(dx[0]) + sqr(dx[1]) + sqr(dx[2]);
                  const auto rinv = rsqrtf(r2);
                  const auto rinv3 = rinv * rinv * rinv;
                  for (int dim = 0; dim < NDIM; dim++) {
                     f[dim][tid] -= dx[dim] * rinv3;
                  }
               }
               __syncthreads();
               for (int P = KICK_BLOCK_SIZE / 2; P >= 1; P /= 2) {
                  if (tid < P) {
                     for (int dim = 0; dim < NDIM; dim++) {
                        f[dim][tid] += f[dim][tid + P];
                     }
                  }
                  __syncthreads();
               }
               if (tid == 0) {
                  for (int dim = 0; dim < NDIM; dim++) {
                     F[dim][k] += f[dim][0];
                  }
               }
               __syncthreads();
            }
         }
      }
   }
}

CUDA_DEVICE
void cuda_pc_interactions(kick_params_type *params_ptr) {
   kick_params_type &params = *params_ptr;
   const int &tid = threadIdx.x;
   __shared__
   extern int shmem_ptr[];
   cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
   auto &f = shmem.f;
   auto &F = shmem.F;
   auto &sinks = shmem.sink;
   auto &inters = params.multi_interactions;
   const auto &myparts = ((tree*) params.tptr)->parts;
   const auto offset = myparts.first;
   const int mmax = ((params.nmulti - 1) / KICK_BLOCK_SIZE + 1) * KICK_BLOCK_SIZE;
   const int nparts = myparts.second - myparts.first;
   for (int i = tid; i < nparts; i += KICK_BLOCK_SIZE) {
      for (int dim = 0; dim < NDIM; dim++) {
         sinks[dim][i] = parts->pos(dim, myparts.first + i);
      }
   }
   for (int i = tid; i < mmax; i += KICK_BLOCK_SIZE) {
      const auto &sources = ((tree*) inters[i])->pos;
      const int nparts = myparts.second - myparts.first;
      for (int k = 0; k < nparts; k++) {
         for (int dim = 0; dim < NDIM; dim++) {
            f[dim][tid] = 0.f;
         }
         __syncthreads();
         if (i < params.nmulti) {
            array<float, NDIM> dx;
            array<float, NDIM + 1> Lforce;
            for (int l = 0; l < NDIM + 1; l++) {
               Lforce[l] = 0.0f;
            }
            for (int dim = 0; dim < NDIM; dim++) {
               dx[dim] = (fixed<int32_t>(sources[dim]) - fixed<int32_t>(sinks[dim][k])).to_float();
            }
            multipole_interaction(Lforce, *((tree*) inters[i])->multi, dx, false);
            for (int dim = 0; dim < NDIM; dim++) {
               f[dim][tid] -= Lforce[dim + 1];
            }
         }
         __syncthreads();
         for (int P = KICK_BLOCK_SIZE / 2; P >= 1; P /= 2) {
            if (tid < P) {
               for (int dim = 0; dim < NDIM; dim++) {
                  f[dim][tid] += f[dim][tid + P];
               }
            }
            __syncthreads();
         }
         if (tid == 0) {
            for (int dim = 0; dim < NDIM; dim++) {
               F[dim][k] += f[dim][0];
            }
         }
         __syncthreads();
      }
   }
}

CUDA_DEVICE kick_return cuda_kick(kick_params_type *params_ptr) {
   kick_params_type &params = *params_ptr;
   __shared__
   extern int shmem_ptr[];
   cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
   tree_ptr tptr = params.tptr;
   const int &tid = threadIdx.x;
   int depth = params.depth;
   kick_return rc;
   auto &F = shmem.F;
   if (((tree*) tptr)->children[0].rank == -1) {
      for (int k = tid; k < MAX_BUCKET_SIZE; k += KICK_BLOCK_SIZE) {
         for (int dim = 0; dim < NDIM; dim++) {
            F[dim][k] = 0.f;
         }
      }
      __syncthreads();
   }
   {
      indices_array &indices = shmem.indices;
      counts_array &count = shmem.count;

      const auto theta2 = params.theta * params.theta;
      array<tree_ptr*, N_INTERACTION_TYPES> all_checks;
      array<int*, N_INTERACTION_TYPES> list_counts;
      all_checks[CC_CP_DIRECT] = params.dstack.get_top_list();
      all_checks[CC_CP_EWALD] = params.estack.get_top_list();
      all_checks[PC_PP_DIRECT] = params.dstack.get_top_list();
      all_checks[PC_PP_EWALD] = params.estack.get_top_list();
      list_counts[CC_CP_DIRECT] = &params.dstack.get_top_count();
      list_counts[CC_CP_EWALD] = &params.estack.get_top_count();
      list_counts[PC_PP_DIRECT] = &params.dstack.get_top_count();
      list_counts[PC_PP_EWALD] = &params.estack.get_top_count();
      array<array<tree_ptr, WORKSPACE_SIZE>*, NITERS> lists;
      auto &multis = params.multi_interactions;
      auto &parti = params.part_interactions;
      auto &next_checks = params.next_checks;
      auto &opened_checks = params.opened_checks;
      lists[MI] = &multis;
      lists[PI] = &parti;
      lists[CI] = &next_checks;
      lists[OI] = &opened_checks;
      const auto &myradius = ((tree*) tptr)->radius;
      const auto &mypos = ((tree*) tptr)->pos;
      int ninteractions = ((tree*) tptr)->children[0].rank == -1 ? 4 : 2;
      for (int type = 0; type < ninteractions; type++) {
         auto *checks = (all_checks[type]);
         const bool ewald_dist = type == PC_PP_EWALD || type == CC_CP_EWALD;
         const bool direct = type == PC_PP_EWALD || type == PC_PP_DIRECT;
         if (tid < NITERS) {
            count[tid] = 0;
         }
         __syncthreads();
         int check_count;
         do {
            check_count = (*list_counts[type]);
            if (check_count) {
               const int cimax = ((check_count - 1) / KICK_BLOCK_SIZE + 1) * KICK_BLOCK_SIZE;
               for (int ci = tid; ci < cimax; ci += KICK_BLOCK_SIZE) {
                  for (int i = 0; i < NITERS; i++) {
                     indices[i][tid + 1] = 0;
                  }
                  __syncthreads();
                  if (tid < NITERS) {
                     indices[tid][0] = 0;
                  }
                  __syncthreads();
                  int list_index = -1;
                  if (ci < check_count) {
                     auto &check = checks[ci];
                     const auto &other_radius = ((const tree*) check)->radius;
                     const auto &other_pos = ((const tree*) check)->pos;
                     float d2 = 0.f;
                     const float R2 = sqr(other_radius + myradius);
                     for (int dim = 0; dim < NDIM; dim++) {
                        d2 += sqr(fixed<int32_t>(other_pos[dim]) - fixed<int32_t>(mypos[dim])).to_float();
                     }
                     if (ewald_dist) {
                        d2 = fmaxf(d2, EWALD_MIN_DIST2);
                     }
                     const bool far = R2 < theta2 * d2;
                     const bool isleaf = ((const tree*) check)->children[0].rank == -1;
                     list_index = int(!far) * (1 + int(isleaf) + int(isleaf && bool(check.opened++)));
                     indices[list_index][tid + 1] = 1;
                  }
                  __syncthreads();
                  for (int P = 1; P < KICK_BLOCK_SIZE; P *= 2) {
                     array<int, NITERS> tmp;
                     if (tid + 1 > P) {
                        for (int i = 0; i < NITERS; i++) {
                           tmp[i] = indices[i][tid - P + 1];
                        }
                     }
                     __syncthreads();
                     if (tid + 1 > P) {
                        for (int i = 0; i < NITERS; i++) {
                           indices[i][tid + 1] += tmp[i];
                        }
                     }
                     __syncthreads();
                  }
                  __syncthreads();
                  if (ci < check_count) {
                     const auto &check = checks[ci];
                     assert(count[list_index] + indices[list_index][tid] >= 0);
                     (*lists[list_index])[count[list_index] + indices[list_index][tid]] = check;
                  }
                  __syncthreads();
                  if (tid < NITERS) {
                     count[tid] += indices[tid][KICK_BLOCK_SIZE];
                  }
                  __syncthreads();
               }
               __syncthreads();
               check_count = 2 * count[CI];
               (type == CC_CP_DIRECT || type == PC_PP_DIRECT) ?
                     params.dstack.resize_top(check_count) : params.estack.resize_top(check_count);
               for (int i = tid; i < count[CI]; i += KICK_BLOCK_SIZE) {
                  const auto children = next_checks[i].get_children();
                  for (int j = 0; j < NCHILD; j++) {
                     checks[2 * i + j] = children[j];
                  }
               }
               __syncthreads();
               if (type == CC_CP_DIRECT || type == CC_CP_EWALD) {
                  check_count += count[OI];
                  (type == CC_CP_DIRECT || type == PC_PP_DIRECT) ?
                        params.dstack.resize_top(check_count) : params.estack.resize_top(check_count);
                  for (int i = tid; i < count[OI]; i += KICK_BLOCK_SIZE) {
                     checks[2 * count[CI] + i] = opened_checks[i];
                  }
               } else {
                  for (int i = tid; i < count[OI]; i += KICK_BLOCK_SIZE) {
                     parti[count[PI] + i] = opened_checks[i];
                  }
               }
               __syncthreads();
               if (tid == 0) {
                  count[CI] = 0;
                  count[OI] = 0;
               }
               __syncthreads();
            }
         } while (direct && check_count);
         if (tid == 0) {
            params.nmulti = count[MI];
            params.npart = count[PI];
         }
         __syncthreads();
         switch (type) {
         case PC_PP_DIRECT:
            //          printf( "%li %li\n", multis.size(), parti.size());
            cuda_pc_interactions(params_ptr);
            cuda_pp_interactions(params_ptr);
            break;
         case CC_CP_DIRECT:
            cuda_cc_interactions(params_ptr);
            cuda_cp_interactions(params_ptr);
            break;
         case PC_PP_EWALD:
            break;
         case CC_CP_EWALD:
            if (count[CI])
               printf("Ewald\n");
            break;
         }

         /*********** DO INTERACTIONS *********************/

      }
   }
   if (!(((tree*) tptr)->children[0].rank == -1)) {
      params.dstack.copy_top();
      params.estack.copy_top();
      if (tid == 0) {
         params.tptr = ((tree*) tptr)->children[LEFT];
         params.depth++;
      }
      __syncthreads();
      kick_return rc1 = cuda_kick(params_ptr);
      params.dstack.pop();
      params.estack.pop();
      if (tid == 0) {
         params.tptr = ((tree*) tptr)->children[RIGHT];
      }
      __syncthreads();
      kick_return rc2 = cuda_kick(params_ptr);
      if (tid == 0) {
         params.depth--;
      }
      __syncthreads();
      rc.rung = max(rc1.rung, rc2.rung);
   } else {
      rc.rung = 0;
   }
   return rc;
}

CUDA_KERNEL cuda_set_kick_params_kernel(particle_set *p, ewald_indices *four_indices, ewald_indices *real_indices,
      periodic_parts *periodic_parts) {
   if (threadIdx.x == 0) {
      parts = p;
      four_indices_ptr = four_indices;
      real_indices_ptr = real_indices;
      periodic_parts_ptr = periodic_parts;
   }
}

void tree::cuda_set_kick_params(particle_set *p, ewald_indices *four_indices, ewald_indices *real_indices,
      periodic_parts *parts) {
cuda_set_kick_params_kernel<<<1,1>>>(p,real_indices, four_indices, parts);
            CUDA_CHECK(cudaDeviceSynchronize());
}

CUDA_KERNEL cuda_kick_kernel(kick_return *res, kick_params_type **params) {
   const int &bid = blockIdx.x;
   res[bid] = cuda_kick(params[bid]);

}

thread_local static std::stack<std::pair<cudaStream_t, cudaEvent_t>> streams;

std::pair<cudaStream_t, cudaEvent_t> get_stream() {
   if (streams.empty()) {
      cudaStream_t stream;
      cudaEvent_t event;
      CUDA_CHECK(cudaStreamCreate(&stream));
      CUDA_CHECK(cudaEventCreate(&event));
      streams.push(std::make_pair(stream, event));
   }
   auto stream = streams.top();
   streams.pop();
   return stream;
}

void cleanup_stream(std::pair<cudaStream_t, cudaEvent_t> s) {
   streams.push(s);
}

CUDA_KERNEL cuda_ewald_cc_kernel(kick_params_type **params_ptr) {
   const int &bid = blockIdx.x;
   cuda_ewald_cc_interactions(params_ptr[bid], true);
}

std::function<bool()> cuda_execute_ewald_kernel(kick_params_type **params_ptr, int grid_size) {
   auto stream = get_stream();
cuda_ewald_cc_kernel<<<grid_size,KICK_BLOCK_SIZE,sizeof(cuda_ewald_shmem),stream.first>>>(params_ptr);
      CUDA_CHECK(cudaEventRecord(stream.second, stream.first));

   struct cuda_ewald_future_shared {
      std::pair<cudaStream_t, cudaEvent_t> stream;
      int grid_size;
      mutable bool ready;
   public:
      cuda_ewald_future_shared() {
         ready = false;
      }
      bool operator()() const {
         if (!ready) {
            if (cudaEventQuery(stream.second) == cudaSuccess) {
               ready = true;
               CUDA_CHECK(cudaStreamSynchronize(stream.first));
               cleanup_stream(stream);
            }
         }
         return ready;
      }
   };

   cuda_ewald_future_shared fut;
   fut.stream = stream;
   fut.grid_size = grid_size;
   std::function < bool() > ready_func = [fut]() {
      return fut();
   };
   return ready_func;
}

std::pair<std::function<bool()>, kick_return*> cuda_execute_kick_kernel(kick_params_type **params, int grid_size) {
   auto stream = get_stream();
   const size_t shmemsize = sizeof(cuda_kick_shmem);
   kick_return *returns;
   CUDA_MALLOC(returns, grid_size);
   /***************************************************************************************************************************************************/
   /**/cuda_kick_kernel<<<grid_size, KICK_BLOCK_SIZE, shmemsize, stream.first>>>(returns,params);/**/
   /**/CUDA_CHECK(cudaEventRecord(stream.second, stream.first));/*******************************************************************************************************/
   /***************************************************************************************************************************************************/

   struct cuda_kick_future_shared {
      std::pair<cudaStream_t, cudaEvent_t> stream;
      kick_return *returns;
      int grid_size;
      mutable bool ready;
   public:
      cuda_kick_future_shared() {
         ready = false;
      }
      bool operator()() const {
         if (!ready) {
            if (cudaEventQuery(stream.second) == cudaSuccess) {
               ready = true;
               CUDA_CHECK(cudaStreamSynchronize(stream.first));
               cleanup_stream(stream);
            }
         }
         return ready;
      }
   };

   cuda_kick_future_shared fut;
   fut.returns = returns;
   fut.stream = stream;
   fut.grid_size = grid_size;
   std::function < bool() > ready_func = [fut]() {
      return fut();
   };
   return std::make_pair(std::move(ready_func), std::move(fut.returns));
}

