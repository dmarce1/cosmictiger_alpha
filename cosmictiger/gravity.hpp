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

CUDA_DEVICE void cuda_cc_interactions(particle_set *parts, const vector<tree_ptr>&, kick_params_type *params_ptr);
#ifdef __CUDACC__
CUDA_DEVICE void cuda_ewald_cc_interactions(particle_set *parts, kick_params_type *params_ptr,
       array<float, KICK_BLOCK_SIZE>  *  lptr);
#endif
CUDA_DEVICE void cuda_cp_interactions(particle_set *parts, const vector<tree_ptr>&, kick_params_type *params_ptr);
CUDA_DEVICE void cuda_pp_interactions(particle_set *parts, const vector<tree_ptr>&, kick_params_type *params_ptr);
#ifdef TEST_FORCE
void cuda_compare_with_direct(particle_set *parts);
#endif

CUDA_DEVICE
void cuda_pc_interactions(particle_set *parts, const vector<tree_ptr>&, kick_params_type *params_ptr);

CUDA_EXPORT inline float distance(fixed32 a, fixed32 b) {
   return  (fixed<int32_t>(a)- fixed<int32_t>(b)).to_float();
}


#define PHI0 -4.375


