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

enum eval_type {
	DIRECT, EWALD
};


CUDA_DEVICE void cuda_cc_interactions(kick_params_type *params_ptr, eval_type);
CUDA_DEVICE void cuda_cp_interactions(kick_params_type *params_ptr);
CUDA_DEVICE void cuda_pp_interactions(kick_params_type *params_ptr, int);
CUDA_DEVICE void cuda_pc_interactions(kick_params_type *params_ptr, int);
CUDA_DEVICE int compress_sinks(kick_params_type *params_ptr);


#ifdef TEST_FORCE
void cuda_compare_with_direct(particle_set parts);
#endif

CUDA_EXPORT inline float distance(fixed32 a, fixed32 b) {
#ifdef PERIODIC_OFF
	return a.to_float() - b.to_float();
#else
	return (fixed<int32_t>(a) - fixed<int32_t>(b)).to_float();
#endif
}

simd_float inline distance(const simd_int& a, const simd_int& b) {
#ifdef PERIODIC_OFF
	simd_float d;
	for (int k = 0; k < simd_float::size(); k++) {
		d[k] = (float(uint32_t(a[k])) - float(uint32_t(b[k]))) * fixed2float;
	}
	return d;
#else
	return simd_float(a - b) * simd_float(fixed2float);
#endif
}

#define PHI0 (-35.0f/16.0f)

