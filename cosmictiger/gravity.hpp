/*
 * gravity.hpp
 *
 *  Created on: Feb 10, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_GRAVITY_HPP_
#define COSMICTIGER_GRAVITY_HPP_


#include <cosmictiger/tree.hpp>


#endif /* COSMICTIGER_GRAVITY_HPP_ */


CUDA_DEVICE void cuda_cc_interactions(particle_set* parts, kick_params_type *params_ptr);
CUDA_DEVICE int cuda_ewald_cc_interactions(particle_set* parts, kick_params_type *params_ptr);
CUDA_DEVICE void cuda_cp_interactions(particle_set* parts, kick_params_type *params_ptr);
CUDA_DEVICE void cuda_pp_interactions(particle_set* parts, kick_params_type *params_ptr);

CUDA_DEVICE
void cuda_pc_interactions(particle_set* parts, kick_params_type *params_ptr);
