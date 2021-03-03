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
#include <cosmictiger/gravity.hpp>

//CUDA_KERNEL cuda_kick()

#define CC_CP_DIRECT 0
#define CC_CP_EWALD 1
#define PC_PP_DIRECT 2
#define PC_PP_EWALD 3
#define N_INTERACTION_TYPES 4

CUDA_DEVICE particle_set *parts;

__managed__ double pp_interaction_time;
__managed__ double pc_interaction_time;
__managed__ double cp_interaction_time;
__managed__ double cc_interaction_time;
__managed__ double total_time;

#define MI 0
#define CI 1
#define OI 2
#define PI 3

CUDA_DEVICE kick_return
cuda_kick(kick_params_type * params_ptr)
{
   kick_params_type &params = *params_ptr;
   __shared__ volatile
   extern int shmem_ptr[];
   cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
   //  printf( "%i\n", params_ptr->depth);
   //   if( params_ptr->depth > TREE_MAX_DEPTH || params_ptr->depth < 0 ) {
//      printf( "%li\n", params_ptr->depth);
   //  }
   tree_ptr tptr = params.tptr;
   tree& me = *((tree*) tptr);
   const int &tid = threadIdx.x;
   kick_return rc;
   auto &F = params.F;
   auto &L = params.L[params.depth];
   if( tid == 0 ) {
      const auto &Lpos = params.Lpos[params.depth];
      array<float, NDIM> dx;
      for (int dim = 0; dim < NDIM; dim++) {
         const auto x1 = me.pos[dim];
         const auto x2 = Lpos[dim];
         dx[dim] = distance(x1, x2);
      }
      shift_expansion(L,dx);
   }

#ifdef COUNT_FLOPS
   int flops = 0;
#endif
   if (((tree*) tptr)->children[0].ptr == 0) {
      for (int k = tid; k < MAX_BUCKET_SIZE; k += KICK_BLOCK_SIZE) {
         for (int dim = 0; dim < NDIM; dim++) {
            F[dim][k] = 0.f;
         }
      }
      __syncwarp();
   }
   {
      auto &indices = shmem.indices;
      auto &count = shmem.count;

      const auto theta2 = params.theta * params.theta;
      array<vector<tree_ptr>*, NITERS> lists;
      auto &multis = params.multi_interactions;
      auto &parti = params.part_interactions;
      auto &next_checks = params.next_checks;
      auto &opened_checks = params.opened_checks;
      lists[MI] = &multis;
      lists[PI] = &parti;
      lists[CI] = &next_checks;
      lists[OI] = &opened_checks;
      for( int i = 0; i < NITERS; i++) {
         lists[i]->resize(0);
      }
      const auto myradius = SINK_BIAS * ((tree*) tptr)->radius;
      const auto &mypos = ((tree*) tptr)->pos;
      const bool iamleaf = ((tree*) tptr)->children[0].ptr;
      int ninteractions = iamleaf == 0 ? 4 : 2;
      for (int type = 0; type < ninteractions; type++) {
         const bool ewald_dist = type == PC_PP_EWALD || type == CC_CP_EWALD;
         auto& checks = ewald_dist ? params.echecks : params.dchecks;
         const bool direct = type == PC_PP_EWALD || type == PC_PP_DIRECT;
         if (tid < NITERS) {
            count[tid] = 0;
         }
         for( int i = 0; i < NITERS; i++) {
            lists[i]->resize(0);
         }
         __syncwarp();
         int check_count;
         do {
            check_count = checks.size();
            flops += check_count * FLOPS_OPEN;
            if (check_count) {
               const int cimax = ((check_count - 1) / KICK_BLOCK_SIZE + 1) * KICK_BLOCK_SIZE;
               for (int ci = tid; ci < cimax; ci += KICK_BLOCK_SIZE) {
                  for (int i = 0; i < NITERS; i++) {
                     indices[i][tid + 1] = 0;
                  }
                  __syncwarp();
                  if (tid < NITERS) {
                     indices[tid][0] = 0;
                  }
                  __syncwarp();
                  int list_index;
                  const auto h = params.hsoft;
                  if (ci < check_count) {
                     auto &check = checks[ci];
                     const auto &other_radius = ((const tree*) check)->radius;
                     const auto &other_pos = ((const tree*) check)->pos;
                     float d2 = 0.f;
                     const float R2 = sqr(other_radius + myradius + h);                 // 2
#ifdef PERIODIC_OFF
                     for (int dim = 0; dim < NDIM; dim++) {                         // 3
                        d2 += sqr(other_pos[dim].to_float() - mypos[dim].to_float());
                     }
#else
                     for (int dim = 0; dim < NDIM; dim++) {                         // 3
                        d2 += sqr(distance(other_pos[dim], mypos[dim]));
                     }
#endif
                     if (ewald_dist) {
                        d2 = fmaxf(d2, EWALD_MIN_DIST2);                            // 1
                     }
                     const bool far = R2 < theta2 * d2;                             // 2
                     const bool far2 = sqr(other_radius*params.theta + myradius + h) < theta2 * d2;
                     const bool far3 = sqr(other_radius + myradius*params.theta/SINK_BIAS + h) < theta2 * d2;
                     //     const bool isleaf = ((const tree*) check)->children[0].ptr == 0;
                     const bool isleaf = ((const tree*) check)->parts.second - ((const tree*) check)->parts.first <= GROUP_SIZE;
                     auto& other_opened = check.opened;
                     const auto& me_opened = direct;
                     if( !me_opened && far && !other_opened ) {
                        list_index = MI;                                               //CC
                     } else if ( me_opened && far3 && !other_opened ) {
                        list_index = MI;                                               //PC
                     } else if( !me_opened && far2 && isleaf && !iamleaf) {
                        list_index = PI;                                               //CP
                     } else  if(  me_opened && other_opened ) {
                        if( far3 ) {
                           list_index = MI;
                        } else {
                           list_index = PI;
                        }
                     } else if( isleaf ) {
                        other_opened++;
                        list_index = OI;
                     } else {
                        list_index = CI;
                     }
                     indices[list_index][tid + 1] = 1;
                  }
                  __syncwarp();
                  for (int P = 1; P < KICK_BLOCK_SIZE; P *= 2) {
                     array<int, NITERS> tmp;
                     if (tid - P + 1 >= 0) {
                        for (int i = 0; i < NITERS; i++) {
                           tmp[i] = indices[i][tid - P + 1];
                        }
                     }
                     __syncwarp();
                     if (tid - P + 1 >= 0) {
                        for (int i = 0; i < NITERS; i++) {
                           indices[i][tid + 1] += tmp[i];
                        }
                     }
                     __syncwarp();
                  }
                  __syncwarp();
                  for( int i = 0; i < NITERS; i++) {
                     assert(indices[i][tid] <= indices[i][tid+1] );
                     lists[i]->resize(count[i]+indices[i][KICK_BLOCK_SIZE]);
                  }
                  if( ci < check_count ) {
                     const auto &check = checks[ci];
                     assert(count[list_index] + indices[list_index][tid] >= 0);
                     //          printf( "%i %i\n",(*lists[list_index]).size(), count[list_index] + indices[list_index][tid] );
                     (*lists[list_index])[count[list_index] + indices[list_index][tid]] = check;
                  }
                  __syncwarp();
                  if (tid < NITERS) {
                     count[tid] += indices[tid][KICK_BLOCK_SIZE];
                  }
                  __syncwarp();
               }
               __syncwarp();
               check_count = 2 * count[CI];
               checks.resize(check_count);
               for (int i = tid; i < count[CI]; i += KICK_BLOCK_SIZE) {
                  const auto children = next_checks[i].get_children();
                  for (int j = 0; j < NCHILD; j++) {
                     checks[2 * i + j] = children[j];
                  }
               }
               __syncwarp();
               check_count += count[OI];
               checks.resize(check_count);
               for (int i = tid; i < count[OI]; i += KICK_BLOCK_SIZE) {
                  checks[2 * count[CI] + i] = opened_checks[i];
               }
               __syncwarp();
               if (tid == 0) {
                  count[CI] = 0;
                  count[OI] = 0;
               }
               __syncwarp();
               next_checks.resize(0);
               opened_checks.resize(0);
            }
         }while (direct && check_count);
//         if( tid == 0 ) {
//            printf( "%i %i %i\n", params.depth, count[MI], count[PI]);
//         }
         __syncwarp();
         switch (type) {
            case PC_PP_DIRECT:
            //          printf( "%li %li\n", multis.size(), parti.size());
            flops += cuda_pc_interactions(parts,multis, params_ptr);
            flops += cuda_pp_interactions(parts,parti, params_ptr);
            //          if( tid == 0 ) {
            //                          printf( "%i %i %i\n", params_ptr->depth, parti.size(), multis.size());
//            }
            break;
            case CC_CP_DIRECT:
            flops += cuda_cc_interactions(parts,multis, params_ptr);
            flops += cuda_cp_interactions(parts,parti,params_ptr);
            //          if( tid == 0 ) {
            //                        printf( "%i %i %i\n", params_ptr->depth, parti.size(), multis.size());
            //         }
            break;

            case PC_PP_EWALD:
            if(count[PI] > 0 ) {
               //        printf( "PP Ewald should not exist\n");
               //  __trap();
            }
            if(count[MI] > 0 ) {
               //      printf( "PC Ewald should not exist\n");
               //   __trap();
            }
            break;
            case CC_CP_EWALD:
#ifndef PERIODIC_OFF
            if(count[PI] > 0 ) {
               printf( "CP Ewald should not exist\n");
               //     __trap();
            }
            if( count[MI] > 0 ) {
               flops += cuda_ewald_cc_interactions(parts,params_ptr, &shmem.Lreduce);
            }
#endif
            break;
         }
      }
      rc.flops = flops;
   }
   if (!(((tree*) tptr)->children[0].ptr == 0)) {
      params.dchecks.push_top();
      params.echecks.push_top();
      if (tid == 0) {
         params.depth++;
         params.L[params.depth] = L;
         params.Lpos[params.depth] = me.pos;
         params.tptr = ((tree*) tptr)->children[RIGHT];
      }
      __syncwarp();
      kick_return rc1 = cuda_kick(params_ptr);
      if (tid == 0) {
         params.L[params.depth] = L;
         params.tptr = ((tree*) tptr)->children[LEFT];
      }
      __syncwarp();
      params.dchecks.pop_top();
      params.echecks.pop_top();
      kick_return rc2 = cuda_kick(params_ptr);
      if (tid == 0) {
         params.depth--;
      }
      __syncwarp();
      rc.rung = max(rc1.rung, rc2.rung);
      rc.flops += rc1.flops + rc2.flops;
      //   printf( "%li\n", rc.flops);
   } else {
      auto& rungs = shmem.rungs;
      rungs[tid] = 0;
      const auto& myparts = ((tree*)params.tptr)->parts;
      const auto invlog2 = 1.0f / logf(2);
      for (int k = tid; k < myparts.second - myparts.first; k += KICK_BLOCK_SIZE) {
         const auto this_rung = parts->rung(k+myparts.first);
         if( this_rung >= params.rung ) {

            array<float,NDIM> g;
            float phi;
            array<float,NDIM> dx;
            for (int dim = 0; dim < NDIM; dim++) {
               const auto x2 = me.pos[dim];
               const auto x1 = parts->pos(dim,k+myparts.first);
               dx[dim] = distance(x1, x2);
            }
            shift_expansion(L, g, phi, dx);
            for (int dim = 0; dim < NDIM; dim++) {
               F[dim][k] += g[dim];
            }
#ifdef TEST_FORCE
            for( int dim = 0; dim < NDIM; dim++) {
               parts->force(dim,k+myparts.first) = F[dim][k];
            }
#endif
            float dt = params.t0 / (1<<this_rung);
            for( int dim = 0; dim < NDIM; dim++) {
               parts->vel(dim,k+myparts.first) += 0.5 * dt * F[dim][k];
            }
            float fmag = 0.0;
            for( int dim = 0; dim < NDIM; dim++) {
               fmag += sqr(F[dim][k]);
            }
            fmag = sqrtf(fmag);
            //   printf( "%e\n", fmag);
            assert(fmag > 0.0);
            dt = fminf(sqrt(params.scale * params.eta / fmag), params.t0);
            int new_rung = fmaxf(fmaxf(ceil(logf(params.t0/dt) * invlog2), this_rung-1),params.rung);
            dt = params.t0 / (1<<new_rung);
            for( int dim = 0; dim < NDIM; dim++) {
               parts->vel(dim,k+myparts.first) += 0.5 * dt * F[dim][k];
            }
            rungs[tid] = fmaxf(rungs[tid],new_rung);
            parts->set_rung(new_rung, k+myparts.first);
         }
      }
      __syncwarp();
      for( int P = KICK_BLOCK_SIZE/2; P>=1; P/=2) {
         if( tid < P) {
            rungs[tid] = fmaxf(rungs[tid], rungs[tid+P]);
         }
         __syncwarp();
      }
      rc.rung = rungs[0];
   }
   return rc;
}

CUDA_KERNEL cuda_set_kick_params_kernel(particle_set *p, ewald_indices *real_indices, ewald_indices *four_indices,
      periodic_parts *periodic_parts) {
   if (threadIdx.x == 0) {
      parts = p;
      four_indices_ptr = four_indices;
      real_indices_ptr = real_indices;
      periodic_parts_ptr = periodic_parts;
      expansion_init();
      pp_interaction_time = pc_interaction_time = cp_interaction_time = cc_interaction_time = 0.0;

   }
}
void tree::cuda_set_kick_params(particle_set *p, ewald_indices *real_indices, ewald_indices *four_indices,
      periodic_parts *parts) {
cuda_set_kick_params_kernel<<<1,1>>>(p,real_indices, four_indices, parts);
                                                                                          CUDA_CHECK(cudaDeviceSynchronize());
}

#ifdef TIMINGS
extern __managed__ double pp_crit1_time;
extern __managed__ double pp_crit2_time;
#endif

CUDA_KERNEL cuda_kick_kernel(kick_return *res, kick_params_type *params) {
   const int &bid = blockIdx.x;
#ifdef TIMINGS
   auto tm = clock64();
#endif
   res[bid] = cuda_kick(params + bid);
   __syncwarp();
   if (threadIdx.x == 0) {
      //     printf( "Kick done\n");
      params[bid].kick_params_type::~kick_params_type();
      //   printf("%e %e %e %e %e\n", walk_time/total_time, pp_interaction_time/total_time, pc_interaction_time/total_time, cp_interaction_time/total_time, cc_interaction_time/total_time);
   }

}

thread_local static std::stack<cudaStream_t> streams;

cudaStream_t get_stream() {
   if (streams.empty()) {
      cudaStream_t stream;
      CUDA_CHECK(cudaStreamCreate(&stream));
      streams.push(stream);
   }
   auto stream = streams.top();
   streams.pop();
   return stream;
}

void cleanup_stream(cudaStream_t s) {
   streams.push(s);
}

CUDA_KERNEL cuda_ewald_cc_kernel(kick_params_type **params_ptr) {
   __shared__
   volatile
   extern int shmem_ptr[];
   cuda_ewald_shmem &shmem = *((cuda_ewald_shmem*) (shmem_ptr));
   const int &bid = blockIdx.x;
   auto pptr = params_ptr[bid];
   auto rc = cuda_ewald_cc_interactions(parts, pptr, &shmem.Lreduce);
   __syncwarp();
   if (threadIdx.x == 0) {
      params_ptr[bid]->flops = rc;
   }
}

std::function<bool()> cuda_execute_ewald_kernel(kick_params_type **params_ptr, int grid_size) {
   auto stream = get_stream();
   /***/cuda_ewald_cc_kernel<<<grid_size,KICK_BLOCK_SIZE,sizeof(cuda_ewald_shmem),stream>>>(params_ptr);

   struct cuda_ewald_future_shared {
      cudaStream_t stream;
      int grid_size;
      mutable bool ready;
   public:
      cuda_ewald_future_shared() {
         ready = false;
      }
      bool operator()() const {
         if (!ready) {
            if (cudaStreamQuery(stream) == cudaSuccess) {
               ready = true;
               CUDA_CHECK(cudaStreamSynchronize(stream));
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

std::pair<std::function<bool()>, kick_return*> cuda_execute_kick_kernel(kick_params_type *params, int grid_size,
      cudaStream_t stream) {
   const size_t shmemsize = sizeof(cuda_kick_shmem);
   unified_allocator alloc;
   kick_return *returns = (kick_return*) alloc.allocate(grid_size * sizeof(kick_return));
// printf( "a\n");
//  CUDA_MALLOC(returns, grid_size);
// printf( "b\n");
//  printf( "Shmem = %li\n", shmemsize);
   /***************************************************************************************************************************************************/
   /**/cuda_kick_kernel<<<grid_size, KICK_BLOCK_SIZE, shmemsize, stream>>>(returns,params);/**/
   /***************************************************************************************************************************************************/
// printf( "c\n");
   struct cuda_kick_future_shared {
      cudaStream_t stream;
      kick_return *returns;
      int grid_size;
      mutable bool ready;
   public:
      cuda_kick_future_shared() {
         ready = false;
      }
      bool operator()() const {
         if (!ready) {
            if (cudaStreamQuery(stream) == cudaSuccess) {
               ready = true;
               CUDA_CHECK(cudaStreamSynchronize(stream));
               cleanup_stream(stream);
            }
         }
         return ready;
      }
   };
// printf( "d\n");

   cuda_kick_future_shared fut;
   fut.returns = returns;
   fut.stream = stream;
   fut.grid_size = grid_size;
   std::function < bool() > ready_func = [fut]() {
      return fut();
   };
// printf( "e\n");

   return std::make_pair(std::move(ready_func), std::move(fut.returns));
}

