/*
 * expansion.cpp
 *
 *  Created on: Feb 10, 2021
 *      Author: dmarce1
 */

#include <cosmictiger/expansion.hpp>
#include <cosmictiger/ewald_indices.hpp>
#include <cosmictiger/array.hpp>
#include <cosmictiger/code_gen.hpp>


__device__ expansion<float> Lfactor_gpu;
expansion<float> Lfactor_cpu;

__device__ void expansion_init() {
}

__host__ void expansion_init_cpu() {
}

__constant__ int offsets4[6] = { 0, 1, 1, 2, 2, 2 };
__constant__ int offsets5[6] = { 0, 2, 2, 4, 4, 4 };
__constant__ int offsets10[10] = { 0, 1, 1, 2, 2, 2, 3, 3, 3, 3 };

CUDA_DEVICE expansion<float>& cuda_shift_expansion(expansion<float> &L, const array<float, NDIM> &dX, bool do_phi) {
	const int& tid = threadIdx.x;
	if (tid == 0) {
		L = expansion_translate<float>(L, dX);
	}
	__syncwarp();

	return L;
}

expansion<float>& shift_expansion(expansion<float> &L, const array<float, NDIM> &dX, bool do_phi) {
	L = expansion_translate<float>(L, dX);
	return L;
}

CUDA_EXPORT void shift_expansion(const expansion<float> &L0, array<float, NDIM> &g, float &phi,
		const array<float, NDIM> &dX, bool do_phi) {
	const auto L = expansion_translate<float>(L0, dX);
	phi = L(0, 0, 0);
	g[0] = -L(1, 0, 0);
	g[1] = -L(0, 1, 0);
	g[2] = -L(0, 0, 1);
}
