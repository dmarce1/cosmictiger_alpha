/*
 * expansion.cpp
 *
 *  Created on: Feb 10, 2021
 *      Author: dmarce1
 */

#include <cosmictiger/expansion.hpp>
#include <cosmictiger/array.hpp>

__device__ expansion<float> Lfactor_gpu;
expansion<float> Lfactor_cpu;

__device__ void expansion_init() {
}

__host__ void expansion_init_cpu() {
}


CUDA_DEVICE expansion<float>& cuda_shift_expansion(expansion<float> &L, const array<float, NDIM> &dX, bool do_phi) {
	return L;
}

expansion<float>& shift_expansion(expansion<float> &L, const array<float, NDIM> &dX, bool do_phi) {
	return L;
}

CUDA_EXPORT void shift_expansion(const expansion<float> &L, array<float, NDIM> &g, float &phi,
		const array<float, NDIM> &dX, bool do_phi) {

}
