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

#define MI 0
#define CI 1
#define OI 2
#define PI 3

CUDA_DEVICE kick_return
cuda_kick(kick_params_type * params_ptr)
{
   kick_params_type &params = *params_ptr;
   __shared__
   extern int shmem_ptr[];
   cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
   tree_ptr tptr = params.tptr;
   tree& me = *((tree*) tptr);
   const int &tid = threadIdx.x;
   kick_return rc;
   auto &F = shmem.F;
   auto &L = params.L[params.depth];
   if( tid == 0 ) {
      const auto &Lpos = params.Lpos[params.depth];
      array<accum_real, NDIM> dx;
      for (int dim = 0; dim < NDIM; dim++) {
#ifdef ACCUMULATE_DOUBLE_PRECISION
         const auto x1 = me.pos[dim].to_double();
         const auto x2 = Lpos[dim].to_double();
#else
         const auto x1 = me.pos[dim].to_float();
         const auto x2 = Lpos[dim].to_float();
#endif
         dx[dim] = x1 - x2;
      }
      shift_expansion(L,dx);
   }

#ifdef COUNT_FLOPS
   auto& flops = shmem.flops;
   flops[tid] = 0;
#endif
   if (((tree*) tptr)->children[0].rank == -1) {
      for (int k = tid; k < MAX_BUCKET_SIZE; k += KICK_BLOCK_SIZE) {
         for (int dim = 0; dim < NDIM; dim++) {
            F[dim][k] = 0.f;
         }
      }
      __syncthreads();
   }
   {
      auto &indices = shmem.indices;
      auto &count = shmem.count;

      const auto theta2 = params.theta * params.theta;
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
         const bool ewald_dist = type == PC_PP_EWALD || type == CC_CP_EWALD;
         auto& checks = ewald_dist ? params.echecks[params.depth] : params.dchecks[params.depth];
         const bool direct = type == PC_PP_EWALD || type == PC_PP_DIRECT;
         if (tid < NITERS) {
            count[tid] = 0;
         }
         __syncthreads();
         int check_count;
         do {
            check_count = checks.size();
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
                     const float R2 = sqr(other_radius + myradius);                 // 2
                     for (int dim = 0; dim < NDIM; dim++) {                         // 3
                        d2 += sqr(fixed<int32_t>(other_pos[dim]) - fixed<int32_t>(mypos[dim])).to_float();
                     }
                     if (ewald_dist) {
                        d2 = fmaxf(d2, EWALD_MIN_DIST2);                            // 1
                        flops[tid]++;
                     }
                     const bool far = R2 < theta2 * d2;                             // 2
                     flops[tid] += 6;
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
               checks.resize(check_count);
               for (int i = tid; i < count[CI]; i += KICK_BLOCK_SIZE) {
                  const auto children = next_checks[i].get_children();
                  for (int j = 0; j < NCHILD; j++) {
                     checks[2 * i + j] = children[j];
                  }
               }
               __syncthreads();
               if (type == CC_CP_DIRECT || type == CC_CP_EWALD) {
                  check_count += count[OI];
                  checks.resize(check_count);
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
         }while (direct && check_count);
         if (tid == 0) {
            params.nmulti = count[MI];
            params.npart = count[PI];
         }
         __syncthreads();
         switch (type) {
            case PC_PP_DIRECT:
            //          printf( "%li %li\n", multis.size(), parti.size());
            cuda_pc_interactions(parts,params_ptr);
            cuda_pp_interactions(parts,params_ptr);
            break;
            case CC_CP_DIRECT:
            cuda_cc_interactions(parts,params_ptr);
            cuda_cp_interactions(parts,params_ptr);
            break;
            case PC_PP_EWALD:
            if(count[PI] > 0 ) {
             //  printf( "PP Ewald should not exist\n");
             //  __trap();
            }
            if(count[MI] > 0 ) {
            //   printf( "PC Ewald should not exist\n");
            //   __trap();
            }
            break;
            case CC_CP_EWALD:
            if(count[PI] > 0 ) {
          //     printf( "CP Ewald should not exist\n");
          //     __trap();
            }
            if( count[MI] > 0 ) {
               float old_flops = flops[tid];
               int cuda_flops = cuda_ewald_cc_interactions(parts,params_ptr);
               flops[tid] = old_flops;
               flops[0] += cuda_flops;
            }
            break;
         }
      }
#ifdef COUNT_FLOPS
      __syncthreads();
      for( int P = KICK_BLOCK_SIZE/2; P >= 1; P/=2) {
         if( tid < P ) {
            flops[tid] += flops[tid+P];
         }
         __syncthreads();
      }
      rc.flops = flops[0];
#endif
   }
   if (!(((tree*) tptr)->children[0].rank == -1)) {
      params.dchecks[params.depth+1] = params.dchecks[params.depth];
      params.echecks[params.depth+1] = params.echecks[params.depth];
      if (tid == 0) {
         params.depth++;
         params.L[params.depth] = L;
         params.Lpos[params.depth] = me.pos;
         params.tptr = ((tree*) tptr)->children[LEFT];
      }
      __syncthreads();
      kick_return rc1 = cuda_kick(params_ptr);
      params.dchecks[params.depth].swap(params.dchecks[params.depth-1]);
      params.echecks[params.depth].swap(params.echecks[params.depth-1]);
      if (tid == 0) {
         params.L[params.depth] = L;
         params.tptr = ((tree*) tptr)->children[RIGHT];
      }
      __syncthreads();
      kick_return rc2 = cuda_kick(params_ptr);
      if (tid == 0) {
         params.depth--;
      }
      __syncthreads();
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
#ifndef TEST_FORCE
         if( this_rung >= params.rung ) {
#endif

            array<accum_real,NDIM> g;
            accum_real phi;
            array<accum_real,NDIM> dx;
            for (int dim = 0; dim < NDIM; dim++) {
#ifdef ACCUMULATE_DOUBLE_PRECISION
               const auto x2 = me.pos[dim].to_double();
               const auto x1 = parts->pos(dim,k+myparts.first).to_double();
#else
               const auto x2 = me.pos[dim].to_float();
               const auto x1 = parts->pos(dim,k+myparts.first).to_float();
#endif
               dx[dim] = x1 - x2;
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
#ifndef TEST_FORCE
         }
#endif
      }
      __syncthreads();
      for( int P = KICK_BLOCK_SIZE/2; P>=1; P/=2) {
         if( tid < P) {
            rungs[tid] = fmaxf(rungs[tid], rungs[tid+P]);
         }
         __syncthreads();
      }
      rc.rung = rungs[0];
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
      expansion_init();
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
   __syncthreads();
   if( threadIdx.x == 0 ) {
      params[bid]->kick_params_type::~kick_params_type();
   }


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

   auto rc = cuda_ewald_cc_interactions(parts, params_ptr[bid]);
   __syncthreads();
   if (threadIdx.x == 0) {
      params_ptr[bid]->flops = rc;
   }
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

std::pair<std::function<bool()>, kick_return*> cuda_execute_kick_kernel(kick_params_type **params, int grid_size,std::pair<cudaStream_t,cudaEvent_t> stream) {
   const size_t shmemsize = sizeof(cuda_kick_shmem);
   kick_return * returns;
  // printf( "a\n");
   CUDA_MALLOC(returns, grid_size);
  // printf( "b\n");
   /***************************************************************************************************************************************************/
   /**/cuda_kick_kernel<<<grid_size, KICK_BLOCK_SIZE, shmemsize, stream.first>>>(returns,params);/**/
 //  /**/CUDA_CHECK(cudaEventRecord(stream.second, stream.first));/*******************************************************************************************************/
   /***************************************************************************************************************************************************/
  // printf( "c\n");
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
            if (cudaStreamQuery(stream.first) == cudaSuccess) {
               ready = true;
 //              CUDA_CHECK(cudaStreamSynchronize(stream.first));
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

