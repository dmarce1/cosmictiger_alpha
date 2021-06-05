/*
 * expansion.cpp
 *
 *  Created on: Feb 10, 2021
 *      Author: dmarce1
 */

#include <cosmictiger/expansion.hpp>
#include <cosmictiger/array.hpp>


__device__ void expansion_init() {
}

__host__ void expansion_init_cpu() {
}

CUDA_DEVICE expansion<float>& cuda_shift_expansion(expansion<float> &L, const array<float, NDIM> &dX, bool do_phi) {
	if( threadIdx.x == 0 ) {
//		translate_expansion(L, L, dX[0], dX[1], dX[2]);
	}
	return L;
}

expansion<float>& shift_expansion(expansion<float> &L, const array<float, NDIM> &dX, bool do_phi) {
	//translate_expansion(L, L, dX[0], dX[1], dX[2]);
	return L;
}

CUDA_EXPORT void shift_expansion(const expansion<float> &Lin, array<float, NDIM> &g, float &phi,
		const array<float, NDIM> &dX, bool do_phi) {
	sphericalY<float, 2> Lout;
	Lout = Lin;
//	translate_expansion(Lout, Lin, dX[0], dX[1], dX[2]);
	phi += Lout(0).real();
	g[0] += Lout(1, 1).real() / 0.707106781;
	g[1] += Lout(1, 1).imag() / 0.707106781;
	g[2] -= Lout(1).real();
}
