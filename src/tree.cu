#include <cosmictiger/tree.hpp>
#include <cosmictiger/containers.hpp>
#include <cosmictiger/global.hpp>

#include <cstdint>
#include <nvfunctional>

#define KICKBLOCKSIZE 32

#define DIRECT 0
#define EWALD 1

#define CLOUD 0
#define PARTICLE 1

#define LEFT 0
#define RIGHT 1

CUDA_DEVICE
float theta;

CUDA_DEVICE
int kern_cnt;

CUDA_DEVICE
int max_kern_cnt;

CUDA_DEVICE
int min_rung;

CUDA_DEVICE
nvstd::function<kick_return()> tree_kick_child(kick_params *params, int call_depth, int child_index);

template<class T, int N>
CUDA_DEVICE
T reduce_count(array<T,N+1>& count) {
   const int& tid = threadIdx.x;
   assert(blocDim.x==N+1);
   if( tid == 0 ) {
      count[0] = 0;
   }
   CUDA_SYNC();
   for(int P = 1; P < N; P*=2) {
      T tmp;
      if( P - tid>= 0 ) {
         tmp = count[P-tid+1];
      } else {
         tmp = T(0);
      }
      CUDA_SYNC();
      count[tid+1] += tmp;
      CUDA_SYNC();
   }
   return count[N-1];
}

CUDA_DEVICE
kick_return tree_kick(kick_params *params, int call_depth) {
   CUDA_SHARED array<int8_t, KICKBLOCKSIZE + 1>
   mcount;
   CUDA_SHARED array<int8_t, KICKBLOCKSIZE + 1>
   ccount;
   CUDA_SHARED array<int8_t, KICKBLOCKSIZE + 1>
   pcount;
   const int &tid = threadIdx.x;
   kick_return rc;
   params->call_stack.resize(call_depth + 1);
   CUDA_SYNC();
   auto &multi_i = params->multi_i;
   auto &parts_i = params->part_i;
   const tree *tptr = (tree*) params->call_stack[call_depth].tptr;
   const float theta2 = theta * theta;
   constexpr float edist2 = EWALD_DISTANCE * EWALD_DISTANCE;

#pragma loop unroll
   for (int TYPE = CLOUD; TYPE <= PARTICLE; TYPE++) {
#pragma loop unroll
      for (int I = DIRECT; I <= EWALD; I++) { /* BEGIN CC and CP  */
         auto &checks = (I == DIRECT) ? params->call_stack[call_depth].dchecks : params->call_stack[call_depth].echecks;
         auto &next_checks =
               (I == DIRECT) ? params->call_stack[call_depth + 1].dchecks : params->call_stack[call_depth + 1].echecks;
         while (TYPE == PARTICLE && checks.size()) {
            next_checks.reserve(2 * checks.size());
            next_checks.resize(0);
            multi_i.reserve(checks.size());
            parts_i.reserve(checks.size());
            multi_i.resize(0);
            parts_i.resize(0);
            CUDA_SYNC();
            const int end = ((checks.size() - 1) / KICKBLOCKSIZE) * KICKBLOCKSIZE + 1;
            for (int i = tid; i < end; i += KICKBLOCKSIZE) {
               mcount[tid + 1] = ccount[tid + 1] = pcount[tid + 1] = 0;
               tree_client multi;
               tree_client parts;
               array<tree_client, NCHILD> child_checks;
               const auto* check = (tree*) checks[i];
               if (i < checks.size()) {
                  float dist2 = 0.f;
                  for (int dim = 0; dim < NDIM; dim++) {
                     dist2 += sqr(fixed<int32_t>(tptr->pos[dim]) - fixed<int32_t>(check->pos[dim])).to_float();
                  }
                  if (I == EWALD) {
                     dist2 = fmaxf(dist2, (edist2));
                  }
                  const float R2 = sqr(tptr->radius + check->radius);
                  if (R2 < theta2 * dist2) {
                     mcount[tid + 1] = 1;
                     multi = checks[i];
                  } else {
                     if (!check->leaf()) {
                        ccount[tid + 1] = 1;
                        child_checks = check->children;
                     } else {
                        pcount[tid + 1] = 1;
                        parts = checks[i];
                     }
                  }
               }
               const int mtot = reduce_count<int8_t, KICKBLOCKSIZE>(mcount);
               const int ctot = reduce_count<int8_t, KICKBLOCKSIZE>(pcount);
               const int ptot = reduce_count<int8_t, KICKBLOCKSIZE>(ccount);
               const int moffset = multi_i.size();
               const int coffset = next_checks.size();
               const int poffset = parts_i.size();
               multi_i.resize(moffset + mtot);
               parts_i.resize(poffset + ptot);
               next_checks.resize(coffset + 2 * ctot);
               CUDA_SYNC();
               if (mcount[tid + 1] != mcount[tid]) {
                  multi_i[moffset + tid] = multi;
               } else if (pcount[tid + 1] != pcount[tid]) {
                  parts_i[poffset + tid] = parts;
               } else {
                  assert(ccount[tid + 1] != ccount[tid]);
                  for (int ci = 0; ci < NCHILD; ci++) {
                     next_checks[coffset + 2 * tid + ci] = child_checks[ci];
                  }
               }
            }
            checks.swap(next_checks);
         }

         /*** DO INTERACTIONS ***/
         if (TYPE == CLOUD) {
            if (I == DIRECT) {
               /**direct CC and CP**/
            } else {
               assert(I == EWALD);
               /**ewald CC and CP**/
            }
         } else {
            assert(TYPE == PARTICLE);
            if (I == DIRECT) {
               /**direct PC and PP**/
            } else {
               assert(I == EWALD);
               /**ewald PC and PP**/
            }
         }

      } /* END */
   }
   if (!tptr->leaf()) {
      array<nvstd::function<kick_return()>, NCHILD> futs;
      params->call_stack[call_depth + 1].dchecks = params->call_stack[call_depth].dchecks;
      params->call_stack[call_depth + 1].echecks = params->call_stack[call_depth].echecks;
      params->call_stack[call_depth + 1].tptr = tptr->children[LEFT];
      futs[LEFT] = tree_kick_child(params, call_depth + 1, LEFT);
      params->call_stack[call_depth + 1].dchecks.swap(params->call_stack[call_depth].dchecks);
      params->call_stack[call_depth + 1].echecks.swap(params->call_stack[call_depth].echecks);
      params->call_stack[call_depth + 1].tptr = tptr->children[RIGHT];
      futs[RIGHT] = tree_kick_child(params, call_depth + 1, RIGHT);
      rc.rung = 0;
      for (int ci = 0; ci < NCHILD; ci++) {
         const auto tmp = futs[ci]();
         rc.rung = max(tmp.rung, rc.rung);
      }
   }
   return rc;
}

CUDA_KERNEL tree_kick_parameters_kernel(int kcnt, int rung, float theta_) {
   if (threadIdx.x == 0) {
      max_kern_cnt = kcnt;
      kern_cnt = 0;
      min_rung = rung;
      theta = theta_;
   }
}

CUDA_KERNEL tree_kick_kernel(kick_return *rcptr, kick_params *params) {
   kick_return rc = tree_kick(params, 0);
   if (threadIdx.x == 0) {
      *rcptr = std::move(rc);
   }
}

CUDA_DEVICE nvstd::function<kick_return()>
tree_kick_child(kick_params *params, int call_depth, int child_index) {
const int &tid = threadIdx.x;
CUDA_SHARED
int cnt;
bool launch = true;
if (child_index == LEFT) {
   if (tid == 0) {
      cnt = atomicAdd(&kern_cnt, 1);
   }
   CUDA_SYNC();
   launch = cnt < kern_cnt;
}
if (!launch) {
   if (tid == 0) {
      atomicAdd(&kern_cnt, -1);
   }
   CUDA_SYNC();
   kick_return rc = tree_kick(params, call_depth);
   return [rc]() {
      return rc;
   };
} else {
   kick_return *rcptr;
   kick_params *pptr;
   if (tid == 0) {
      CUDA_MALLOC(rcptr, 1);CUDA_MALLOC(pptr, 1);new (rcptr) kick_return();
      new (pptr) kick_params();
      pptr->call_stack.resize(1, params->call_stack[call_depth]);
      tree_kick_kernel<<<1,KICKBLOCKSIZE>>>(rcptr,pptr);

      CUDA_SYNC();
   }
   return [=]() {
      CUDA_SHARED kick_return
      rc;
      if (tid == 0) {
         CUDA_CHECK(cudaDeviceSynchronize());atomicAdd(&kern_cnt, -1);
         rc = std::move(*rcptr);
         rcptr->~kick_return();
         pptr->~kick_params();
         CUDA_FREE(rcptr);CUDA_FREE(pptr);}
CUDA_SYNC();
return rc;
};
}
}

kick_return tree::kick(tree_client root_ptr, int rung, float theta) {
kick_return *rcptr;
kick_params *pptr;
kick_params root_params;
kick_return rc;
CUDA_MALLOC(rcptr, 1);
CUDA_MALLOC(pptr, 1);
new (rcptr) kick_return();
new (pptr) kick_params();
root_params.call_stack.resize(1);
root_params.call_stack[0].dchecks.push_back(root_ptr);
root_params.call_stack[0].echecks.push_back(root_ptr);
root_params.call_stack[0].L = 0.0;
for (int dim = 0; dim < NDIM; dim++) {
root_params.call_stack[0].Lcom[dim] = 0.5;
}
*pptr = std::move(root_params);
const int kcnt = global().cuda.devices[0].multiProcessorCount - 1;
/***************************************************************************/
/**/tree_kick_parameters_kernel <<< 1, 1 >>> (kcnt, rung, theta);/*****************/
/**/tree_kick_kernel<<<1,KICKBLOCKSIZE>>>(rcptr,pptr);/*********************/
/**/CUDA_CHECK(cudaDeviceSynchronize());/***********************************/
/***************************************************************************/
rc = std::move(*rcptr);
rcptr->~kick_return();
pptr->~kick_params();
CUDA_FREE(rcptr);
CUDA_FREE(pptr);
return rc;

}
