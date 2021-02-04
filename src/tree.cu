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

CUDA_DEVICE
void reduce_indexes(array<int, KICK_BLOCK_SIZE + 1> &counts) {
   const int &tid = threadIdx.x;
   counts[0] = 0;
   CUDA_SYNC();
   for (int P = 1; P < KICK_BLOCK_SIZE; P *= 2) {
      int tmp;
      if (tid - P + 1 >= 0) {
         tmp = counts[tid - P + 1];
      }
      CUDA_SYNC();
      if (tid - P + 1 >= 0) {
         counts[tid + 1] += tmp;
      }
      CUDA_SYNC();
   }
}

CUDA_DEVICE kick_return cuda_kick(tree_ptr tptr, kick_stack &stacks, kick_workspace_t &workspace, int depth) {
   const int &tid = threadIdx.x;
   kick_return rc;
   __shared__ array<int, KICK_BLOCK_SIZE + 1>
   mindices;
   __shared__ array<int, KICK_BLOCK_SIZE + 1>
   cindices;
   __shared__ array<int, KICK_BLOCK_SIZE + 1>
   pindices;
   const auto theta2 = theta * theta;
   array<checks_type*, N_INTERACTION_TYPES> all_checks;
   all_checks[CC_CP_DIRECT] = &stacks.dchecks[depth];
   all_checks[CC_CP_EWALD] = &stacks.echecks[depth];
   all_checks[PC_PP_DIRECT] = &stacks.dchecks[depth];
   all_checks[PC_PP_EWALD] = &stacks.echecks[depth];
   auto &multis = workspace.multi_interactions;
   auto &parti = workspace.part_interactions;
   auto &next_checks = workspace.next_checks;
   int ninteractions = ((tree*) tptr)->children[0] == tree_ptr() ? 4 : 2;
   for (int type = 0; type < ninteractions; type++) {
      auto &checks = *(all_checks[type]);
      if (tid == 0) {
         next_checks.resize(0);
         multis.resize(0);
         parti.resize(0);
      }
      CUDA_SYNC();
      const bool ewald_dist = type == PC_PP_EWALD || type == CC_CP_EWALD;
      const bool direct = type == PC_PP_EWALD || type == PC_PP_DIRECT;
      do {
       const int cimax = ((checks.size() - 1) / KICK_BLOCK_SIZE + 1) * KICK_BLOCK_SIZE;
         for (int ci = tid; ci < cimax; ci += KICK_BLOCK_SIZE) {
            mindices[tid + 1] = cindices[tid + 1] = pindices[tid + 1] = 0;
            if (ci < checks.size()) {
               const auto other_radius = ((tree*) checks[ci])->radius;
               const auto other_pos = ((tree*) checks[ci])->pos;
               float d2 = 0.f;
               const float R2 = sqr(other_radius + ((tree*) tptr)->radius);
               for (int dim = 0; dim < NDIM; dim++) {
                  d2 += sqr(fixed<int32_t>(other_pos[dim]) - fixed<int32_t>(((tree*) tptr)->pos[dim])).to_float();
               }
               if (ewald_dist) {
                  d2 = fmaxf(d2, EWALD_MIN_DIST2);
               }
               const bool far = R2 < theta2 * d2;
               if (far) {
                  mindices[tid + 1] = 1;
               } else if (!checks[ci].is_leaf()) {
                  cindices[tid + 1] = 1;
               } else {
                  pindices[tid + 1] = 1;
               }
            }
            reduce_indexes (mindices);
            reduce_indexes (cindices);
            reduce_indexes (pindices);
            int moffset = multis.size();
            int poffset = parti.size();
            int coffset = next_checks.size();
            CUDA_SYNC();
            if (tid == 0) {
               multis.resize(multis.size() + mindices[KICK_BLOCK_SIZE]);
               parti.resize(parti.size() + pindices[KICK_BLOCK_SIZE]);
               next_checks.resize(next_checks.size() + 2 * cindices[KICK_BLOCK_SIZE]);
            }
            CUDA_SYNC();
            if (ci < checks.size()) {
               if (mindices[tid] != mindices[tid + 1]) {
                  assert(mindices[tid] < mindices[tid + 1]);
                  multis[moffset + mindices[tid]] = checks[ci];
               } else if (pindices[tid] != pindices[tid + 1]) {
                  assert(pindices[tid] < pindices[tid + 1]);
                  parti[poffset + pindices[tid]] = checks[ci];
               } else if (cindices[tid] != cindices[tid + 1]) {
                  assert(cindices[tid] < cindices[tid + 1]);
                  next_checks[coffset + 2 * cindices[tid] + LEFT] = checks[ci].get_children()[LEFT];
                  next_checks[coffset + 2 * cindices[tid] + RIGHT] = checks[ci].get_children()[RIGHT];
               }
            }
            CUDA_SYNC();
         }
         CUDA_SYNC();
         if (tid == 0) {
            checks = std::move(next_checks);
         }
         CUDA_SYNC();
      } while (direct && checks.size());

      /*********** DO INTERACTIONS *********************/

   }
   if (!(((tree*) tptr)->children[0] == tree_ptr())) {
      stacks.dchecks[depth + 1] = stacks.dchecks[depth];
      stacks.echecks[depth + 1] = stacks.echecks[depth];
      kick_return rc1 = cuda_kick(((tree*) tptr)->children[LEFT], stacks, workspace, depth + 1);
      stacks.dchecks[depth + 1] = std::move(stacks.dchecks[depth]);
      stacks.echecks[depth + 1] = std::move(stacks.echecks[depth]);
      kick_return rc2 = cuda_kick(((tree*) tptr)->children[RIGHT], stacks, workspace, depth + 1);
      rc.rung = max(rc1.rung, rc2.rung);
   } else {
      rc.rung = 0;
   }
   return rc;
}

CUDA_KERNEL cuda_set_kick_params(float theta_, int rung_) {
   if (threadIdx.x == 0) {
      theta = theta_;
      rung = rung_;
   }
}

CUDA_KERNEL cuda_kick_kernel(finite_vector<kick_return, KICK_GRID_SIZE> *rc,
      finite_vector<kick_stack, KICK_GRID_SIZE> *stacks, finite_vector<tree_ptr, KICK_GRID_SIZE> *roots,
      finite_vector<int, KICK_GRID_SIZE> *depths, finite_vector<kick_workspace_t, KICK_GRID_SIZE> *workspaces) {
   const int &bid = blockIdx.x;

   (*rc)[bid] = cuda_kick((*roots)[bid], (*stacks)[bid], (*workspaces)[bid], (*depths)[bid]);

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
   finite_vector<kick_workspace_t, KICK_GRID_SIZE>* workspaces_ptr;
   CUDA_MALLOC(stacks_ptr, 1);
   CUDA_MALLOC(roots_ptr, 1);
   CUDA_MALLOC(depths_ptr, 1);
   CUDA_MALLOC(workspaces_ptr, 1);
   new (stacks_ptr) finite_vector<kick_stack, KICK_GRID_SIZE>(std::move(stacks));
   new (roots_ptr) finite_vector<tree_ptr, KICK_GRID_SIZE>(std::move(roots));
   new (depths_ptr) finite_vector<int, KICK_GRID_SIZE>(std::move(depths));
   new (workspaces_ptr) finite_vector<kick_workspace_t, KICK_GRID_SIZE>(std::move(workspaces));

cuda_set_kick_params<<<1,1>>>(0.7,0);
   /***************************************************************************************************************************************************/
   /**/cuda_kick_kernel<<<grid_size, KICK_BLOCK_SIZE, 0, stream>>>(rcptr, stacks_ptr,roots_ptr, depths_ptr, workspaces_ptr);/**/
/**/            CUDA_CHECK(cudaEventRecord(event, stream));/*******************************************************************************************************/
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
      finite_vector<kick_workspace_t, KICK_GRID_SIZE>* workspaces_ptr;
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
        //       printf( "Kernel done\n");
               CUDA_CHECK(cudaEventDestroy(event));
               CUDA_CHECK(cudaStreamDestroy(stream));
               *returns = std::move(*rcptr);
               rcptr->finite_vector<kick_return, KICK_GRID_SIZE>::~finite_vector<kick_return, KICK_GRID_SIZE>();
               CUDA_FREE(rcptr);
               stacks_ptr->finite_vector<kick_stack, KICK_GRID_SIZE>::~finite_vector<kick_stack, KICK_GRID_SIZE>();
               roots_ptr->finite_vector<tree_ptr, KICK_GRID_SIZE>::~finite_vector<tree_ptr, KICK_GRID_SIZE>();
               depths_ptr->finite_vector<int, KICK_GRID_SIZE>::~finite_vector<int, KICK_GRID_SIZE>();
               workspaces_ptr->finite_vector<kick_workspace_t, KICK_GRID_SIZE>::~finite_vector<kick_workspace_t, KICK_GRID_SIZE>();
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

