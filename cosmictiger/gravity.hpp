/*
 * gravity.hpp
 *
 *  Created on: Feb 10, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_GRAVITY_HPP_
#define COSMICTIGER_GRAVITY_HPP_

#include <cosmictiger/defs.hpp>
#include <cosmictiger/array.hpp>
#include <cosmictiger/tree.hpp>

#endif /* COSMICTIGER_GRAVITY_HPP_ */

CUDA_DEVICE int cuda_cc_interactions(particle_set *parts, const vector<tree_ptr>&, kick_params_type *params_ptr);
#ifdef __CUDACC__
CUDA_DEVICE int cuda_ewald_cc_interactions(particle_set *parts, kick_params_type *params_ptr,
       array<hifloat, KICK_BLOCK_SIZE>  *  lptr);
#endif
CUDA_DEVICE int cuda_cp_interactions(particle_set *parts, const vector<tree_ptr>&, kick_params_type *params_ptr);
CUDA_DEVICE int cuda_pp_interactions(particle_set *parts, const vector<tree_ptr>&, kick_params_type *params_ptr);
#ifdef TEST_FORCE
void cuda_compare_with_direct(particle_set *parts);
#endif

CUDA_DEVICE
int cuda_pc_interactions(particle_set *parts, const vector<tree_ptr>&, kick_params_type *params_ptr);

CUDA_EXPORT inline float distance(fixed32 a, fixed32 b) {
   const float dif = (a.to_float() - b.to_float());
   const float absdif = fabsf(dif);
   return copysignf(fminf(absdif, 1.f - absdif), dif * (.5f - absdif));
}



