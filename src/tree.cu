#include <cosmictiger/tree.hpp>
#include <cosmictiger/cuda.hpp>
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

struct cuda_kick_params {
   tree_ptr tptr;
   kick_stack &stacks;
   kick_workspace_t &workspace;
   int depth;
   CUDA_EXPORT cuda_kick_params(kick_stack &s, kick_workspace_t &w) :
         stacks(s), workspace(w) {
   }
};

using indices_array = array<array<int8_t, KICK_BLOCK_SIZE + 1>, NITERS>;
using counts_array = array<int16_t, NITERS>;

using pos_array = array<fixed32,WORKSPACE_SIZE>;

CUDA_DEVICE void cuda_pp_interactions(cuda_kick_params &params) {
   const int &tid = threadIdx.x;
   const int &depth = params.depth;
   __shared__
   extern int shmem[];
   pos_array &xsource = *((pos_array*) (shmem));
   pos_array &ysource = *((pos_array*) ((int8_t*) shmem) + sizeof(pos_array));
   pos_array &zsource = *((pos_array*) ((int8_t*) shmem) + 2 * sizeof(pos_array));
   auto &inters = params.workspace.part_interactions;
   if (inters.size()) {
      int i = 0;
      while (i < inters.size()) {
         pair<size_t, size_t> these_parts;
         these_parts = ((tree*) inters[i])->parts;
         i++;
         while (i < inters.size()) {
            if (((tree*) inters[i])->parts.first == these_parts.second) {
               these_parts.second = ((tree*) inters[i])->parts.second;
               i++;
            } else {
               break;
            }
         }
         for (int j = these_parts.first + tid; j < these_parts.second; j += KICK_BLOCK_SIZE) {
            const auto offset = these_parts.first;
            assert(j - offset <= KICK_BLOCK_SIZE);
            xsource[j - offset] = parts->pos(0, j);
            ysource[j - offset] = parts->pos(1, j);
            zsource[j - offset] = parts->pos(2, j);
            printf("Loading\n");
         }
      }
   }
}

CUDA_DEVICE kick_return cuda_kick(cuda_kick_params &params) {
   __shared__
   extern int shmem[];
   kick_stack &stacks = params.stacks;
   tree_ptr tptr = params.tptr;
   const int &tid = threadIdx.x;
   int depth = params.depth;
   kick_return rc;
   {
      indices_array &indices = *(indices_array*) shmem;
      counts_array &count = *(counts_array*) (((uint8_t*) shmem) + sizeof(indices_array));

      kick_workspace_t &workspace = params.workspace;

      const auto theta2 = theta * theta;
      array<checks_type*, N_INTERACTION_TYPES> all_checks;
      all_checks[CC_CP_DIRECT] = &stacks.dchecks[depth];
      all_checks[CC_CP_EWALD] = &stacks.echecks[depth];
      all_checks[PC_PP_DIRECT] = &stacks.dchecks[depth];
      all_checks[PC_PP_EWALD] = &stacks.echecks[depth];
      array<finite_vector<tree_ptr, WORKSPACE_SIZE>*, NITERS> lists;
      auto &multis = workspace.multi_interactions;
      auto &parti = workspace.part_interactions;
      auto &next_checks = workspace.next_checks;
      auto &opened_checks = workspace.opened_checks;
      lists[MI] = &multis;
      lists[PI] = &parti;
      lists[CI] = &next_checks;
      lists[OI] = &opened_checks;
      const auto &myradius = ((tree*) tptr)->radius;
      const auto &mypos = ((tree*) tptr)->pos;
      int ninteractions = ((tree*) tptr)->children[0].rank == -1 ? 4 : 2;
      for (int type = 0; type < ninteractions; type++) {
         for (int i = 0; i < NITERS; i++) {
            lists[i]->resize(WORKSPACE_SIZE);
         }
         auto &checks = *(all_checks[type]);
         const bool ewald_dist = type == PC_PP_EWALD || type == CC_CP_EWALD;
         const bool direct = type == PC_PP_EWALD || type == PC_PP_DIRECT;
         if (tid < NITERS) {
            count[tid] = 0;
         }
         __syncthreads();
         int check_count;
         do {
            if (checks.size()) {
               check_count = checks.size();
               checks.resize(WORKSPACE_SIZE);
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
               for (int i = tid; i < count[CI]; i += KICK_BLOCK_SIZE) {
                  const auto children = next_checks[i].get_children();
                  for (int j = 0; j < NCHILD; j++) {
                     checks[2 * i + j] = children[j];
                  }
               }
               check_count = 2 * count[CI];
               if (type == CC_CP_DIRECT || type == CC_CP_EWALD) {
                  for (int i = tid; i < count[OI]; i += KICK_BLOCK_SIZE) {
                     checks[check_count + i] = opened_checks[i];
                  }
                  check_count += count[OI];
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
               checks.resize(check_count);
            }
         } while (direct && check_count);
         multis.resize(count[MI]);
         parti.resize(count[PI]);

         if (type == PC_PP_DIRECT) {
            //    cuda_pp_interactions(params);
         }

         /*********** DO INTERACTIONS *********************/

      }
   }
   if (!(((tree*) tptr)->children[0].rank == -1)) {
      stacks.dchecks[depth + 1] = stacks.dchecks[depth];
      stacks.echecks[depth + 1] = stacks.echecks[depth];
      params.depth++;
      params.tptr = ((tree*) tptr)->children[LEFT];
      kick_return rc1 = cuda_kick(params);
      stacks.dchecks[depth + 1] = std::move(stacks.dchecks[depth]);
      stacks.echecks[depth + 1] = std::move(stacks.echecks[depth]);
      params.tptr = ((tree*) tptr)->children[RIGHT];
      kick_return rc2 = cuda_kick(params);
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

CUDA_KERNEL cuda_kick_kernel(finite_vector<kick_return, KICK_GRID_SIZE> *rc,
      finite_vector<kick_stack, KICK_GRID_SIZE> *stacks, finite_vector<tree_ptr, KICK_GRID_SIZE> *roots,
      finite_vector<int, KICK_GRID_SIZE> *depths, finite_vector<kick_workspace_t, KICK_GRID_SIZE> *workspaces) {
   const int &bid = blockIdx.x;
   cuda_kick_params params((*stacks)[bid], (*workspaces)[bid]);
   params.tptr = (*roots)[bid];
   params.depth = (*depths)[bid];
   (*rc)[bid] = cuda_kick(params);

}

std::pair<std::function<bool()>, std::shared_ptr<finite_vector<kick_return, KICK_GRID_SIZE>>> cuda_execute_kick_kernel(
      finite_vector<kick_stack, KICK_GRID_SIZE> &&stacks, finite_vector<tree_ptr, KICK_GRID_SIZE> &&roots,
      finite_vector<int, KICK_GRID_SIZE> &&depths, int grid_size) {
   std::vector < std::function < kick_return() >> returns;
   finite_vector<kick_return, KICK_GRID_SIZE> *rcptr;
   CUDA_MALLOC(rcptr, 1);
   new (rcptr) finite_vector<kick_return, KICK_GRID_SIZE>();
   rcptr->resize(grid_size);
   cudaStream_t stream;
   cudaEvent_t event;
   CUDA_CHECK(cudaStreamCreate(&stream));
   CUDA_CHECK(cudaEventCreate(&event));
   finite_vector<kick_workspace_t, KICK_GRID_SIZE> workspaces;
   workspaces.resize(KICK_GRID_SIZE);

   finite_vector<kick_stack, KICK_GRID_SIZE> *stacks_ptr;
   finite_vector<tree_ptr, KICK_GRID_SIZE> *roots_ptr;
   finite_vector<int, KICK_GRID_SIZE> *depths_ptr;
   finite_vector<kick_workspace_t, KICK_GRID_SIZE> *workspaces_ptr;
   CUDA_MALLOC(stacks_ptr, 1);
   CUDA_MALLOC(roots_ptr, 1);
   CUDA_MALLOC(depths_ptr, 1);
   CUDA_MALLOC(workspaces_ptr, 1);
   new (stacks_ptr) finite_vector<kick_stack, KICK_GRID_SIZE>(std::move(stacks));
   new (roots_ptr) finite_vector<tree_ptr, KICK_GRID_SIZE>(std::move(roots));
   new (depths_ptr) finite_vector<int, KICK_GRID_SIZE>(std::move(depths));
   new (workspaces_ptr) finite_vector<kick_workspace_t, KICK_GRID_SIZE>(std::move(workspaces));
   const size_t shmemsize = std::max(sizeof(indices_array) + sizeof(counts_array), NDIM * sizeof(pos_array));
   /***************************************************************************************************************************************************/
   /**/cuda_kick_kernel<<<grid_size, KICK_BLOCK_SIZE, shmemsize, stream>>>(rcptr, stacks_ptr,roots_ptr, depths_ptr, workspaces_ptr);/**/
   /**/   CUDA_CHECK(cudaEventRecord(event, stream));/*******************************************************************************************************/
   /***************************************************************************************************************************************************/

   struct cuda_kick_future_shared {
      cudaStream_t stream;
      cudaEvent_t event;
      std::shared_ptr<finite_vector<kick_return, KICK_GRID_SIZE>> returns;
      finite_vector<kick_return, KICK_GRID_SIZE> *rcptr;
      int grid_size;
      finite_vector<kick_stack, KICK_GRID_SIZE> *stacks_ptr;
      finite_vector<tree_ptr, KICK_GRID_SIZE> *roots_ptr;
      finite_vector<int, KICK_GRID_SIZE> *depths_ptr;
      finite_vector<kick_workspace_t, KICK_GRID_SIZE> *workspaces_ptr;
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
               *returns = std::move(*rcptr);
               rcptr->finite_vector<kick_return, KICK_GRID_SIZE>::~finite_vector<kick_return, KICK_GRID_SIZE>();
               stacks_ptr->finite_vector<kick_stack, KICK_GRID_SIZE>::~finite_vector<kick_stack, KICK_GRID_SIZE>();
               roots_ptr->finite_vector<tree_ptr, KICK_GRID_SIZE>::~finite_vector<tree_ptr, KICK_GRID_SIZE>();
               depths_ptr->finite_vector<int, KICK_GRID_SIZE>::~finite_vector<int, KICK_GRID_SIZE>();
               workspaces_ptr->finite_vector<kick_workspace_t, KICK_GRID_SIZE>::~finite_vector<kick_workspace_t,
               KICK_GRID_SIZE>();
               CUDA_FREE(rcptr);
               CUDA_FREE(stacks_ptr);
               CUDA_FREE(roots_ptr);
               CUDA_FREE(depths_ptr);
               CUDA_FREE(workspaces_ptr);
            }
         }
         return ready;
      }
   };

   cuda_kick_future_shared fut;
   fut.returns = std::make_shared<finite_vector<kick_return, KICK_GRID_SIZE>>();
   fut.stream = stream;
   fut.event = event;
   fut.rcptr = rcptr;
   fut.grid_size = grid_size;
   fut.stacks_ptr = stacks_ptr;
   fut.roots_ptr = roots_ptr;
   fut.depths_ptr = depths_ptr;
   fut.workspaces_ptr = workspaces_ptr;
   std::function < bool() > ready_func = [fut]() {
      return fut();
   };
   return std::make_pair(std::move(ready_func), std::move(fut.returns));
}

