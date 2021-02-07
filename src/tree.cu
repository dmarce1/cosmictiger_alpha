#include <cosmictiger/tree.hpp>
#include <cosmictiger/cuda.hpp>
#include <cosmictiger/interactions.hpp>
#include <functional>

//CUDA_KERNEL cuda_kick()

#define CC_CP_DIRECT 0
#define CC_CP_EWALD 1
#define PC_PP_DIRECT 2
#define PC_PP_EWALD 3
#define N_INTERACTION_TYPES 4

CUDA_DEVICE float theta;
CUDA_DEVICE int8_t rung;
CUDA_DEVICE particle_set *parts;

#define NITERS 4
#define MI 0
#define CI 1
#define OI 2
#define PI 3

using indices_array = array<array<int8_t, KICK_BLOCK_SIZE + 1>, NITERS>;
using counts_array = array<int16_t, NITERS>;

#define KICK_PP_MAX 128
#define KICK_CC_MAX 32

using pos_array = array<fixed32,KICK_PP_MAX>;

#define MAX_BUCKET_SIZE 64
struct cuda_kick_shmem {
   indices_array indices;
   counts_array count;
   array<float, KICK_BLOCK_SIZE> f_x;
   array<float, KICK_BLOCK_SIZE> f_y;
   array<float, KICK_BLOCK_SIZE> f_z;
   array<float, MAX_BUCKET_SIZE> Fx;
   array<float, MAX_BUCKET_SIZE> Fy;
   array<float, MAX_BUCKET_SIZE> Fz;
   array<fixed32, KICK_PP_MAX> x;
   array<fixed32, KICK_PP_MAX> y;
   array<fixed32, KICK_PP_MAX> z;
   array<expansion, KICK_BLOCK_SIZE> L;
};

CUDA_DEVICE void cuda_cc_interactions(kick_params_type *params_ptr) {
   kick_params_type &params = *params_ptr;
   const int &tid = threadIdx.x;
   __shared__
   extern int shmem_ptr[];
   cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
   auto &Lreduce = shmem.L;
   auto &multis = params.multi_interactions;
   for (int i = tid; i < LP; i += KICK_BLOCK_SIZE) {
      Lreduce[tid][i] = 0.0;
   }
   __syncthreads();
   const auto &pos = ((tree*) params.tptr)->pos;
   for (int i = tid; i < params.nmulti; i += KICK_BLOCK_SIZE) {
      const multipole mpole = *((tree*) multis[i])->multi;
      expansion L;
      array<float, NDIM> fpos;
      for (int dim = 0; dim < NDIM; dim++) {
         fpos[dim] = (fixed<int32_t>(params.Lpos[params.depth][dim]) - fixed<int32_t>(pos[dim])).to_float();
      }
      multipole_interaction(L, mpole, fpos, false);
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
   auto &Lreduce = shmem.L;
   auto &source_x = shmem.x;
   auto &source_y = shmem.y;
   auto &source_z = shmem.z;
   auto &inters = params.part_interactions;
   const auto sink_x = params.Lpos[params.depth][0];
   const auto sink_y = params.Lpos[params.depth][1];
   const auto sink_z = params.Lpos[params.depth][2];
   if (params.npart) {
      int i = 0;
      const auto &myparts = ((tree*) params.tptr)->parts;
      __syncthreads();
      while (i < params.npart) {
         pair<size_t, size_t> these_parts;
         these_parts = ((tree*) inters[i])->parts;
         i++;
         while (i < params.npart) {
            auto next_parts = ((tree*) inters[i])->parts;
            if (next_parts.first == these_parts.second && next_parts.second - these_parts.first <= KICK_PP_MAX) {
               these_parts.second = ((tree*) inters[i])->parts.second;
               i++;
            } else {
               break;
            }
         }
         for (int j = these_parts.first + tid; j < these_parts.second; j += KICK_BLOCK_SIZE) {
            const auto j0 = j - these_parts.first;
            source_x[j0] = parts->pos(0, j);
            source_y[j0] = parts->pos(1, j);
            source_z[j0] = parts->pos(2, j);
            array<float, NDIM> dx;
            dx[0] = (fixed<int32_t>(source_x[j0]) - fixed<int32_t>(sink_x)).to_float();
            dx[1] = (fixed<int32_t>(source_y[j0]) - fixed<int32_t>(sink_y)).to_float();
            dx[2] = (fixed<int32_t>(source_z[j0]) - fixed<int32_t>(sink_z)).to_float();
            expansion L;
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
   auto &f_x = shmem.f_x;
   auto &f_y = shmem.f_y;
   auto &f_z = shmem.f_z;
   auto &Fx = shmem.Fx;
   auto &Fy = shmem.Fy;
   auto &Fz = shmem.Fz;
   auto &source_x = shmem.x;
   auto &source_y = shmem.y;
   auto &source_z = shmem.z;
   auto &inters = params.part_interactions;
   if (params.npart) {
      int i = 0;
      const auto &myparts = ((tree*) params.tptr)->parts;
      const size_t nsinks = myparts.second - myparts.first;
      __syncthreads();
      while (i < params.npart) {
         pair<size_t, size_t> these_parts;
         these_parts = ((tree*) inters[i])->parts;
         i++;
         while (i < params.npart) {
            auto next_parts = ((tree*) inters[i])->parts;
            if (next_parts.first == these_parts.second && next_parts.second - these_parts.first <= KICK_PP_MAX) {
               these_parts.second = ((tree*) inters[i])->parts.second;
               i++;
            } else {
               break;
            }
         }
         const auto offset = ((tree*) params.tptr)->parts.first;
         for (int k = 0; k < nsinks; k++) {
            f_x[tid] = f_y[tid] = f_x[tid];
            const auto sink_x = parts->pos(0, offset + k);
            const auto sink_y = parts->pos(1, offset + k);
            const auto sink_z = parts->pos(2, offset + k);
            for (int j = these_parts.first + tid; j < these_parts.second; j += KICK_BLOCK_SIZE) {
               const auto j0 = j - these_parts.first;
               source_x[j0] = parts->pos(0, j);
               source_y[j0] = parts->pos(1, j);
               source_z[j0] = parts->pos(2, j);
               const auto dx = (fixed<int32_t>(source_x[j0]) - fixed<int32_t>(sink_x)).to_float();
               const auto dy = (fixed<int32_t>(source_y[j0]) - fixed<int32_t>(sink_y)).to_float();
               const auto dz = (fixed<int32_t>(source_z[j0]) - fixed<int32_t>(sink_z)).to_float();
               const auto r2 = sqr(dx) + sqr(dy) + sqr(dz);
               const auto rinv = rsqrtf(r2);
               const auto rinv3 = rinv * rinv * rinv;
               f_x[tid] -= dx * rinv3;
               f_y[tid] -= dy * rinv3;
               f_z[tid] -= dz * rinv3;
            }
            __syncthreads();
            for (int P = KICK_BLOCK_SIZE / 2; P >= 1; P /= 2) {
               if (tid < P) {
                  f_x[tid] += f_x[tid + P];
                  f_y[tid] += f_y[tid + P];
                  f_z[tid] += f_z[tid + P];
               }
               __syncthreads();
            }
            if (tid == 0) {
               Fx[k] += f_x[0];
               Fy[k] += f_y[0];
               Fz[k] += f_z[0];
            }
            __syncthreads();
         }
      }
   }
}

CUDA_DEVICE void cuda_pc_interactions(kick_params_type *params_ptr) {
   kick_params_type &params = *params_ptr;
   const int &tid = threadIdx.x;
   __shared__
   extern int shmem_ptr[];
   cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
   auto &f_x = shmem.f_x;
   auto &f_y = shmem.f_y;
   auto &f_z = shmem.f_z;
   auto &Fx = shmem.Fx;
   auto &Fy = shmem.Fy;
   auto &Fz = shmem.Fz;
   auto &inters = params.multi_interactions;
   const auto &myparts = ((tree*) params.tptr)->parts;
   const auto offset = myparts.first;
   const int mmax = ((params.nmulti - 1) / KICK_BLOCK_SIZE + 1) * KICK_BLOCK_SIZE;
   for (int i = tid; i < mmax; i += KICK_BLOCK_SIZE) {
      const auto source_x = ((tree*) inters[i])->pos[0];
      const auto source_y = ((tree*) inters[i])->pos[1];
      const auto source_z = ((tree*) inters[i])->pos[2];
      const int nparts = myparts.second - myparts.first;
      for (int k = 0; k < nparts; k++) {
         f_x[tid] = f_y[tid] = f_x[tid];
         if (i < params.nmulti) {
            const auto sink_x = parts->pos(0, offset + k);
            const auto sink_y = parts->pos(1, offset + k);
            const auto sink_z = parts->pos(2, offset + k);
            array<float, NDIM> dx;
            array<float, NDIM + 1> Lforce;
            for (int l = 0; l < NDIM + 1; l++) {
               Lforce[l] = 0.0f;
            }
            dx[0] = (fixed<int32_t>(source_x) - fixed<int32_t>(sink_x)).to_float();
            dx[1] = (fixed<int32_t>(source_y) - fixed<int32_t>(sink_y)).to_float();
            dx[2] = (fixed<int32_t>(source_z) - fixed<int32_t>(sink_z)).to_float();
            multipole_interaction(Lforce, *((tree*) inters[i])->multi, dx, false);
            f_x[tid] -= Lforce[1];
            f_y[tid] -= Lforce[2];
            f_z[tid] -= Lforce[3];
         }
         __syncthreads();
         for (int P = KICK_BLOCK_SIZE / 2; P >= 1; P /= 2) {
            if (tid < P) {
               f_x[tid] += f_x[tid + P];
               f_y[tid] += f_y[tid + P];
               f_z[tid] += f_z[tid + P];
            }
            __syncthreads();
         }
         if (tid == 0) {
            Fx[k] += f_x[0];
            Fy[k] += f_y[0];
            Fz[k] += f_z[0];
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
   auto &Fx = shmem.Fx;
   auto &Fy = shmem.Fy;
   auto &Fz = shmem.Fz;
   if (((tree*) tptr)->children[0].rank == -1) {
      for (int k = tid; k < MAX_BUCKET_SIZE; k += KICK_BLOCK_SIZE) {
         Fx[k] = 0.f;
         Fy[k] = 0.f;
         Fz[k] = 0.f;
      }
      __syncthreads();
   }
   {
      indices_array &indices = shmem.indices;
      counts_array &count = shmem.count;

      const auto theta2 = theta * theta;
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
         params.nmulti = count[MI];
         params.npart = count[PI];
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
            break;
         }

         /*********** DO INTERACTIONS *********************/

      }
   }
   if (!(((tree*) tptr)->children[0].rank == -1)) {
      params.dstack.copy_top();
      params.estack.copy_top();
      params.tptr = ((tree*) tptr)->children[LEFT];
      params.depth++;
      kick_return rc1 = cuda_kick(params_ptr);
      params.dstack.pop();
      params.estack.pop();
      params.tptr = ((tree*) tptr)->children[RIGHT];
      kick_return rc2 = cuda_kick(params_ptr);
      params.depth--;
      rc.rung = max(rc1.rung, rc2.rung);
   } else {
      rc.rung = 0;
   }
   return rc;
}

CUDA_KERNEL cuda_set_kick_params_kernel(particle_set *p, float theta_, int rung_) {
   if (threadIdx.x == 0) {
      parts = p;
      theta = theta_;
      rung = rung_;
   }
}

void tree::cuda_set_kick_params(particle_set *p, float theta_, int rung_) {
cuda_set_kick_params_kernel<<<1,1>>>(p,theta_,rung_);
                                                                                                                                                               CUDA_CHECK(cudaDeviceSynchronize());
}

CUDA_KERNEL cuda_kick_kernel(kick_return *res, kick_params_type **params) {
   const int &bid = blockIdx.x;
   res[bid] = cuda_kick(params[bid]);

}

std::pair<std::function<bool()>, kick_return*> cuda_execute_kick_kernel(kick_params_type **params, int grid_size) {
   cudaStream_t stream;
   cudaEvent_t event;
   CUDA_CHECK(cudaStreamCreate(&stream));
   CUDA_CHECK(cudaEventCreate(&event));
   //  printf("Shared mem requirements = %li\n", sizeof(cuda_kick_shmem));
   const size_t shmemsize = sizeof(cuda_kick_shmem);
   kick_return *returns;
   CUDA_MALLOC(returns, grid_size);
   /***************************************************************************************************************************************************/
   /**/cuda_kick_kernel<<<grid_size, KICK_BLOCK_SIZE, shmemsize, stream>>>(returns,params);/**/
   /**/   CUDA_CHECK(cudaEventRecord(event, stream));/*******************************************************************************************************/
   /***************************************************************************************************************************************************/

   struct cuda_kick_future_shared {
      cudaStream_t stream;
      cudaEvent_t event;
      kick_return *returns;
      int grid_size;
      mutable bool ready;
   public:
      cuda_kick_future_shared() {
         ready = false;
      }
      bool operator()() const {
         if (!ready) {
            if (cudaEventQuery(event) == cudaSuccess) {
               ready = true;
               CUDA_CHECK(cudaStreamSynchronize(stream));
               CUDA_CHECK(cudaEventDestroy(event));
               CUDA_CHECK(cudaStreamDestroy(stream));
            }
         }
         return ready;
      }
   };

   cuda_kick_future_shared fut;
   fut.returns = returns;
   fut.stream = stream;
   fut.event = event;
   fut.grid_size = grid_size;
   std::function < bool() > ready_func = [fut]() {
      return fut();
   };
   return std::make_pair(std::move(ready_func), std::move(fut.returns));
}

