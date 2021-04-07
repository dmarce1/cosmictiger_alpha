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
	for (int i = 0; i < LP; i++) {
		Lfactor_gpu[i] = float(0.0);
	}
	Lfactor_gpu() += float(1);
	for (int a = 0; a < NDIM; ++a) {
		Lfactor_gpu(a) += float(1.0);
		for (int b = 0; b < NDIM; ++b) {
			Lfactor_gpu(a, b) += float(0.5);
			for (int c = 0; c < NDIM; ++c) {
				Lfactor_gpu(a, b, c) += float(1.0 / 6.0);
				for (int d = 0; d < NDIM; ++d) {
					Lfactor_gpu(a, b, c, d) += float(1.0 / 24.0);
				}
			}
		}
	}
}

__host__ void expansion_init_cpu() {
	for (int i = 0; i < LP; i++) {
		Lfactor_cpu[i] = float(0.0);
	}
	Lfactor_cpu() += float(1);
	for (int a = 0; a < NDIM; ++a) {
		Lfactor_cpu(a) += float(1.0);
		for (int b = 0; b < NDIM; ++b) {
			Lfactor_cpu(a, b) += float(0.5);
			for (int c = 0; c < NDIM; ++c) {
				Lfactor_cpu(a, b, c) += float(1.0 / 6.0);
				for (int d = 0; d < NDIM; ++d) {
					Lfactor_cpu(a, b, c, d) += float(1.0 / 24.0);
				}
			}
		}
	}
	// for( int i = 0; i < 35; i++) {
	// 	printf( "%e\n", 1.0/Lfactor_cpu[i]);
	//}
	//abort();
}

__constant__ int offsets4[6] = { 0, 1, 1, 2, 2, 2 };
__constant__ int offsets5[6] = { 0, 2, 2, 4, 4, 4 };
__constant__ int offsets10[10] = { 0, 1, 1, 2, 2, 2, 3, 3, 3, 3 };

CUDA_DEVICE expansion<float>& cuda_shift_expansion(expansion<float> &L, const array<float, NDIM> &dX, bool do_phi) {
	const int& tid = threadIdx.x;
	float tmp1;
	if (do_phi) {
		float tmp2;
		if (tid == 0) {
			L[0] = fma(L[1], dX[0], L[0]);
			tmp1 = dX[0] * dX[0];
			L[0] = fma(L[4], tmp1 * float(5.000000e-01), L[0]);
			tmp2 = tmp1 * dX[0];
			L[0] = fma(L[10], tmp2 * float(1.666667e-01), L[0]);
			L[0] = fma(L[20], tmp2 * dX[0] * float(4.166667e-02), L[0]);
			L[0] = fma(L[2], dX[1], L[0]);
			tmp1 = dX[1] * dX[0];
			L[0] = fma(L[5], tmp1, L[0]);
			tmp2 = tmp1 * dX[0];
			L[0] = fma(L[11], tmp2 * float(5.000000e-01), L[0]);
			L[0] = fma(L[21], tmp2 * dX[0] * float(1.666667e-01), L[0]);
			tmp1 = dX[1] * dX[1];
			L[0] = fma(L[7], tmp1 * float(5.000000e-01), L[0]);
			tmp2 = tmp1 * dX[0];
			L[0] = fma(L[13], tmp2 * float(5.000000e-01), L[0]);
			L[0] = fma(L[23], tmp2 * dX[0] * float(2.500000e-01), L[0]);
			tmp2 = tmp1 * dX[1];
			L[0] = fma(L[16], tmp2 * float(1.666667e-01), L[0]);
			L[0] = fma(L[26], tmp2 * dX[0] * float(1.666667e-01), L[0]);
			L[0] = fma(L[30], tmp2 * dX[1] * float(4.166667e-02), L[0]);
			L[0] = fma(L[3], dX[2], L[0]);
			tmp1 = dX[2] * dX[0];
			L[0] = fma(L[6], tmp1, L[0]);
			tmp2 = tmp1 * dX[0];
			L[0] = fma(L[12], tmp2 * float(5.000000e-01), L[0]);
			L[0] = fma(L[22], tmp2 * dX[0] * float(1.666667e-01), L[0]);
			tmp1 = dX[2] * dX[1];
			L[0] = fma(L[8], tmp1, L[0]);
			tmp2 = tmp1 * dX[0];
			L[0] = fma(L[14], tmp2, L[0]);
			L[0] = fma(L[24], tmp2 * dX[0] * float(5.000000e-01), L[0]);
			tmp2 = tmp1 * dX[1];
			L[0] = fma(L[17], tmp2 * float(5.000000e-01), L[0]);
			L[0] = fma(L[27], tmp2 * dX[0] * float(5.000000e-01), L[0]);
			L[0] = fma(L[31], tmp2 * dX[1] * float(1.666667e-01), L[0]);
			tmp1 = dX[2] * dX[2];
			L[0] = fma(L[9], tmp1 * float(5.000000e-01), L[0]);
			tmp2 = tmp1 * dX[0];
			L[0] = fma(L[15], tmp2 * float(5.000000e-01), L[0]);
			L[0] = fma(L[25], tmp2 * dX[0] * float(2.500000e-01), L[0]);
			tmp2 = tmp1 * dX[1];
			L[0] = fma(L[18], tmp2 * float(5.000000e-01), L[0]);
			L[0] = fma(L[28], tmp2 * dX[0] * float(5.000000e-01), L[0]);
			L[0] = fma(L[32], tmp2 * dX[1] * float(2.500000e-01), L[0]);
			tmp2 = tmp1 * dX[2];
			L[0] = fma(L[19], tmp2 * float(1.666667e-01), L[0]);
			L[0] = fma(L[29], tmp2 * dX[0] * float(1.666667e-01), L[0]);
			L[0] = fma(L[33], tmp2 * dX[1] * float(1.666667e-01), L[0]);
			L[0] = fma(L[34], tmp2 * dX[2] * float(4.166667e-02), L[0]);
		}
	}
	if (tid == 0) {
		L[1] = fma(L[4], dX[0], L[1]);
		tmp1 = dX[0] * dX[0];
		L[1] = fma(L[10], tmp1 * float(5.000000e-01), L[1]);
		L[1] = fma(L[20], tmp1 * dX[0] * float(1.666667e-01), L[1]);
		L[1] = fma(L[5], dX[1], L[1]);
		tmp1 = dX[1] * dX[0];
		L[1] = fma(L[11], tmp1, L[1]);
		L[1] = fma(L[21], tmp1 * dX[0] * float(5.000000e-01), L[1]);
		tmp1 = dX[1] * dX[1];
		L[1] = fma(L[13], tmp1 * float(5.000000e-01), L[1]);
		L[1] = fma(L[23], tmp1 * dX[0] * float(5.000000e-01), L[1]);
		L[1] = fma(L[26], tmp1 * dX[1] * float(1.666667e-01), L[1]);
		L[1] = fma(L[6], dX[2], L[1]);
		tmp1 = dX[2] * dX[0];
		L[1] = fma(L[12], tmp1, L[1]);
		L[1] = fma(L[22], tmp1 * dX[0] * float(5.000000e-01), L[1]);
		tmp1 = dX[2] * dX[1];
		L[1] = fma(L[14], tmp1, L[1]);
		L[1] = fma(L[24], tmp1 * dX[0], L[1]);
		L[1] = fma(L[27], tmp1 * dX[1] * float(5.000000e-01), L[1]);
		tmp1 = dX[2] * dX[2];
		L[1] = fma(L[15], tmp1 * float(5.000000e-01), L[1]);
		L[1] = fma(L[25], tmp1 * dX[0] * float(5.000000e-01), L[1]);
		L[1] = fma(L[28], tmp1 * dX[1] * float(5.000000e-01), L[1]);
		L[1] = fma(L[29], tmp1 * dX[2] * float(1.666667e-01), L[1]);
		L[2] = fma(L[5], dX[0], L[2]);
		tmp1 = dX[0] * dX[0];
		L[2] = fma(L[11], tmp1 * float(5.000000e-01), L[2]);
		L[2] = fma(L[21], tmp1 * dX[0] * float(1.666667e-01), L[2]);
		L[2] = fma(L[7], dX[1], L[2]);
		tmp1 = dX[1] * dX[0];
		L[2] = fma(L[13], tmp1, L[2]);
		L[2] = fma(L[23], tmp1 * dX[0] * float(5.000000e-01), L[2]);
		tmp1 = dX[1] * dX[1];
		L[2] = fma(L[16], tmp1 * float(5.000000e-01), L[2]);
		L[2] = fma(L[26], tmp1 * dX[0] * float(5.000000e-01), L[2]);
		L[2] = fma(L[30], tmp1 * dX[1] * float(1.666667e-01), L[2]);
		L[2] = fma(L[8], dX[2], L[2]);
		tmp1 = dX[2] * dX[0];
		L[2] = fma(L[14], tmp1, L[2]);
		L[2] = fma(L[24], tmp1 * dX[0] * float(5.000000e-01), L[2]);
		tmp1 = dX[2] * dX[1];
		L[2] = fma(L[17], tmp1, L[2]);
		L[2] = fma(L[27], tmp1 * dX[0], L[2]);
		L[2] = fma(L[31], tmp1 * dX[1] * float(5.000000e-01), L[2]);
		tmp1 = dX[2] * dX[2];
		L[2] = fma(L[18], tmp1 * float(5.000000e-01), L[2]);
		L[2] = fma(L[28], tmp1 * dX[0] * float(5.000000e-01), L[2]);
		L[2] = fma(L[32], tmp1 * dX[1] * float(5.000000e-01), L[2]);
		L[2] = fma(L[33], tmp1 * dX[2] * float(1.666667e-01), L[2]);
		L[3] = fma(L[6], dX[0], L[3]);
		tmp1 = dX[0] * dX[0];
		L[3] = fma(L[12], tmp1 * float(5.000000e-01), L[3]);
		L[3] = fma(L[22], tmp1 * dX[0] * float(1.666667e-01), L[3]);
		L[3] = fma(L[8], dX[1], L[3]);
		tmp1 = dX[1] * dX[0];
		L[3] = fma(L[14], tmp1, L[3]);
		L[3] = fma(L[24], tmp1 * dX[0] * float(5.000000e-01), L[3]);
		tmp1 = dX[1] * dX[1];
		L[3] = fma(L[17], tmp1 * float(5.000000e-01), L[3]);
		L[3] = fma(L[27], tmp1 * dX[0] * float(5.000000e-01), L[3]);
		L[3] = fma(L[31], tmp1 * dX[1] * float(1.666667e-01), L[3]);
		L[3] = fma(L[9], dX[2], L[3]);
		tmp1 = dX[2] * dX[0];
		L[3] = fma(L[15], tmp1, L[3]);
		L[3] = fma(L[25], tmp1 * dX[0] * float(5.000000e-01), L[3]);
		tmp1 = dX[2] * dX[1];
		L[3] = fma(L[18], tmp1, L[3]);
		L[3] = fma(L[28], tmp1 * dX[0], L[3]);
		L[3] = fma(L[32], tmp1 * dX[1] * float(5.000000e-01), L[3]);
		tmp1 = dX[2] * dX[2];
		L[3] = fma(L[19], tmp1 * float(5.000000e-01), L[3]);
		L[3] = fma(L[29], tmp1 * dX[0] * float(5.000000e-01), L[3]);
		L[3] = fma(L[33], tmp1 * dX[1] * float(5.000000e-01), L[3]);
		L[3] = fma(L[34], tmp1 * dX[2] * float(1.666667e-01), L[3]);

	}

	const auto tid4 = tid + 4;
	const auto tid10 = tid + 10;
	__syncwarp();
	if (tid < 6) {
		const auto tidpoff4 = tid + offsets4[tid];
		L[tid4] = fma(L[10 + tid], dX[0], L[tid4]);
		L[tid4] = fma(L[20 + tid], dX[0] * dX[0] * float(5.000000e-01), L[tid4]);
		L[tid4] = fma(L[11 + tidpoff4], dX[1], L[tid4]);
		L[tid4] = fma(L[21 + tidpoff4], dX[1] * dX[0], L[tid4]);
		L[tid4] = fma(L[23 + tid + offsets5[tid]], dX[1] * dX[1] * float(5.000000e-01), L[tid4]);
		L[tid4] = fma(L[12 + tidpoff4], dX[2], L[tid4]);
		L[tid4] = fma(L[22 + tidpoff4], dX[2] * dX[0], L[tid4]);
		L[tid4] = fma(L[24 + tid + offsets5[tid]], dX[2] * dX[1], L[tid4]);
		L[tid4] = fma(L[25 + tid + offsets5[tid]], dX[2] * dX[2] * float(5.000000e-01), L[tid4]);
	}
	__syncwarp();
	if (tid < 10) {
		const auto tidpoff10 = tid + offsets10[tid];
		L[tid10] = fma(L[20 + tid], dX[0], L[tid10]);
		L[tid10] = fma(L[21 + tidpoff10], dX[1], L[tid10]);
		L[tid10] = fma(L[22 + tidpoff10], dX[2], L[tid10]);
	}
	__syncwarp();

	return L;
}

expansion<float>& shift_expansion(expansion<float> &L, const array<float, NDIM> &dX, bool do_phi) {
	float tmp1;
	if (do_phi) {
		float tmp2;
		L[0] = fma(L[1], dX[0], L[0]);
		tmp1 = dX[0] * dX[0];
		L[0] = fma(L[4], tmp1 * float(5.000000e-01), L[0]);
		tmp2 = tmp1 * dX[0];
		L[0] = fma(L[10], tmp2 * float(1.666667e-01), L[0]);
		L[0] = fma(L[20], tmp2 * dX[0] * float(4.166667e-02), L[0]);
		L[0] = fma(L[2], dX[1], L[0]);
		tmp1 = dX[1] * dX[0];
		L[0] = fma(L[5], tmp1, L[0]);
		tmp2 = tmp1 * dX[0];
		L[0] = fma(L[11], tmp2 * float(5.000000e-01), L[0]);
		L[0] = fma(L[21], tmp2 * dX[0] * float(1.666667e-01), L[0]);
		tmp1 = dX[1] * dX[1];
		L[0] = fma(L[7], tmp1 * float(5.000000e-01), L[0]);
		tmp2 = tmp1 * dX[0];
		L[0] = fma(L[13], tmp2 * float(5.000000e-01), L[0]);
		L[0] = fma(L[23], tmp2 * dX[0] * float(2.500000e-01), L[0]);
		tmp2 = tmp1 * dX[1];
		L[0] = fma(L[16], tmp2 * float(1.666667e-01), L[0]);
		L[0] = fma(L[26], tmp2 * dX[0] * float(1.666667e-01), L[0]);
		L[0] = fma(L[30], tmp2 * dX[1] * float(4.166667e-02), L[0]);
		L[0] = fma(L[3], dX[2], L[0]);
		tmp1 = dX[2] * dX[0];
		L[0] = fma(L[6], tmp1, L[0]);
		tmp2 = tmp1 * dX[0];
		L[0] = fma(L[12], tmp2 * float(5.000000e-01), L[0]);
		L[0] = fma(L[22], tmp2 * dX[0] * float(1.666667e-01), L[0]);
		tmp1 = dX[2] * dX[1];
		L[0] = fma(L[8], tmp1, L[0]);
		tmp2 = tmp1 * dX[0];
		L[0] = fma(L[14], tmp2, L[0]);
		L[0] = fma(L[24], tmp2 * dX[0] * float(5.000000e-01), L[0]);
		tmp2 = tmp1 * dX[1];
		L[0] = fma(L[17], tmp2 * float(5.000000e-01), L[0]);
		L[0] = fma(L[27], tmp2 * dX[0] * float(5.000000e-01), L[0]);
		L[0] = fma(L[31], tmp2 * dX[1] * float(1.666667e-01), L[0]);
		tmp1 = dX[2] * dX[2];
		L[0] = fma(L[9], tmp1 * float(5.000000e-01), L[0]);
		tmp2 = tmp1 * dX[0];
		L[0] = fma(L[15], tmp2 * float(5.000000e-01), L[0]);
		L[0] = fma(L[25], tmp2 * dX[0] * float(2.500000e-01), L[0]);
		tmp2 = tmp1 * dX[1];
		L[0] = fma(L[18], tmp2 * float(5.000000e-01), L[0]);
		L[0] = fma(L[28], tmp2 * dX[0] * float(5.000000e-01), L[0]);
		L[0] = fma(L[32], tmp2 * dX[1] * float(2.500000e-01), L[0]);
		tmp2 = tmp1 * dX[2];
		L[0] = fma(L[19], tmp2 * float(1.666667e-01), L[0]);
		L[0] = fma(L[29], tmp2 * dX[0] * float(1.666667e-01), L[0]);
		L[0] = fma(L[33], tmp2 * dX[1] * float(1.666667e-01), L[0]);
		L[0] = fma(L[34], tmp2 * dX[2] * float(4.166667e-02), L[0]);
	}
	L[1] = fma(L[4], dX[0], L[1]);
	tmp1 = dX[0] * dX[0];
	L[1] = fma(L[10], tmp1 * float(5.000000e-01), L[1]);
	L[1] = fma(L[20], tmp1 * dX[0] * float(1.666667e-01), L[1]);
	L[1] = fma(L[5], dX[1], L[1]);
	tmp1 = dX[1] * dX[0];
	L[1] = fma(L[11], tmp1, L[1]);
	L[1] = fma(L[21], tmp1 * dX[0] * float(5.000000e-01), L[1]);
	tmp1 = dX[1] * dX[1];
	L[1] = fma(L[13], tmp1 * float(5.000000e-01), L[1]);
	L[1] = fma(L[23], tmp1 * dX[0] * float(5.000000e-01), L[1]);
	L[1] = fma(L[26], tmp1 * dX[1] * float(1.666667e-01), L[1]);
	L[1] = fma(L[6], dX[2], L[1]);
	tmp1 = dX[2] * dX[0];
	L[1] = fma(L[12], tmp1, L[1]);
	L[1] = fma(L[22], tmp1 * dX[0] * float(5.000000e-01), L[1]);
	tmp1 = dX[2] * dX[1];
	L[1] = fma(L[14], tmp1, L[1]);
	L[1] = fma(L[24], tmp1 * dX[0], L[1]);
	L[1] = fma(L[27], tmp1 * dX[1] * float(5.000000e-01), L[1]);
	tmp1 = dX[2] * dX[2];
	L[1] = fma(L[15], tmp1 * float(5.000000e-01), L[1]);
	L[1] = fma(L[25], tmp1 * dX[0] * float(5.000000e-01), L[1]);
	L[1] = fma(L[28], tmp1 * dX[1] * float(5.000000e-01), L[1]);
	L[1] = fma(L[29], tmp1 * dX[2] * float(1.666667e-01), L[1]);
	L[2] = fma(L[5], dX[0], L[2]);
	tmp1 = dX[0] * dX[0];
	L[2] = fma(L[11], tmp1 * float(5.000000e-01), L[2]);
	L[2] = fma(L[21], tmp1 * dX[0] * float(1.666667e-01), L[2]);
	L[2] = fma(L[7], dX[1], L[2]);
	tmp1 = dX[1] * dX[0];
	L[2] = fma(L[13], tmp1, L[2]);
	L[2] = fma(L[23], tmp1 * dX[0] * float(5.000000e-01), L[2]);
	tmp1 = dX[1] * dX[1];
	L[2] = fma(L[16], tmp1 * float(5.000000e-01), L[2]);
	L[2] = fma(L[26], tmp1 * dX[0] * float(5.000000e-01), L[2]);
	L[2] = fma(L[30], tmp1 * dX[1] * float(1.666667e-01), L[2]);
	L[2] = fma(L[8], dX[2], L[2]);
	tmp1 = dX[2] * dX[0];
	L[2] = fma(L[14], tmp1, L[2]);
	L[2] = fma(L[24], tmp1 * dX[0] * float(5.000000e-01), L[2]);
	tmp1 = dX[2] * dX[1];
	L[2] = fma(L[17], tmp1, L[2]);
	L[2] = fma(L[27], tmp1 * dX[0], L[2]);
	L[2] = fma(L[31], tmp1 * dX[1] * float(5.000000e-01), L[2]);
	tmp1 = dX[2] * dX[2];
	L[2] = fma(L[18], tmp1 * float(5.000000e-01), L[2]);
	L[2] = fma(L[28], tmp1 * dX[0] * float(5.000000e-01), L[2]);
	L[2] = fma(L[32], tmp1 * dX[1] * float(5.000000e-01), L[2]);
	L[2] = fma(L[33], tmp1 * dX[2] * float(1.666667e-01), L[2]);
	L[3] = fma(L[6], dX[0], L[3]);
	tmp1 = dX[0] * dX[0];
	L[3] = fma(L[12], tmp1 * float(5.000000e-01), L[3]);
	L[3] = fma(L[22], tmp1 * dX[0] * float(1.666667e-01), L[3]);
	L[3] = fma(L[8], dX[1], L[3]);
	tmp1 = dX[1] * dX[0];
	L[3] = fma(L[14], tmp1, L[3]);
	L[3] = fma(L[24], tmp1 * dX[0] * float(5.000000e-01), L[3]);
	tmp1 = dX[1] * dX[1];
	L[3] = fma(L[17], tmp1 * float(5.000000e-01), L[3]);
	L[3] = fma(L[27], tmp1 * dX[0] * float(5.000000e-01), L[3]);
	L[3] = fma(L[31], tmp1 * dX[1] * float(1.666667e-01), L[3]);
	L[3] = fma(L[9], dX[2], L[3]);
	tmp1 = dX[2] * dX[0];
	L[3] = fma(L[15], tmp1, L[3]);
	L[3] = fma(L[25], tmp1 * dX[0] * float(5.000000e-01), L[3]);
	tmp1 = dX[2] * dX[1];
	L[3] = fma(L[18], tmp1, L[3]);
	L[3] = fma(L[28], tmp1 * dX[0], L[3]);
	L[3] = fma(L[32], tmp1 * dX[1] * float(5.000000e-01), L[3]);
	tmp1 = dX[2] * dX[2];
	L[3] = fma(L[19], tmp1 * float(5.000000e-01), L[3]);
	L[3] = fma(L[29], tmp1 * dX[0] * float(5.000000e-01), L[3]);
	L[3] = fma(L[33], tmp1 * dX[1] * float(5.000000e-01), L[3]);
	L[3] = fma(L[34], tmp1 * dX[2] * float(1.666667e-01), L[3]);
	L[4] = fma(L[10], dX[0], L[4]);
	L[4] = fma(L[20], dX[0] * dX[0] * float(5.000000e-01), L[4]);
	L[4] = fma(L[11], dX[1], L[4]);
	L[4] = fma(L[21], dX[1] * dX[0], L[4]);
	L[4] = fma(L[23], dX[1] * dX[1] * float(5.000000e-01), L[4]);
	L[4] = fma(L[12], dX[2], L[4]);
	L[4] = fma(L[22], dX[2] * dX[0], L[4]);
	L[4] = fma(L[24], dX[2] * dX[1], L[4]);
	L[4] = fma(L[25], dX[2] * dX[2] * float(5.000000e-01), L[4]);
	L[5] = fma(L[11], dX[0], L[5]);
	L[5] = fma(L[21], dX[0] * dX[0] * float(5.000000e-01), L[5]);
	L[5] = fma(L[13], dX[1], L[5]);
	L[5] = fma(L[23], dX[1] * dX[0], L[5]);
	L[5] = fma(L[26], dX[1] * dX[1] * float(5.000000e-01), L[5]);
	L[5] = fma(L[14], dX[2], L[5]);
	L[5] = fma(L[24], dX[2] * dX[0], L[5]);
	L[5] = fma(L[27], dX[2] * dX[1], L[5]);
	L[5] = fma(L[28], dX[2] * dX[2] * float(5.000000e-01), L[5]);
	L[7] = fma(L[13], dX[0], L[7]);
	L[7] = fma(L[23], dX[0] * dX[0] * float(5.000000e-01), L[7]);
	L[7] = fma(L[16], dX[1], L[7]);
	L[7] = fma(L[26], dX[1] * dX[0], L[7]);
	L[7] = fma(L[30], dX[1] * dX[1] * float(5.000000e-01), L[7]);
	L[7] = fma(L[17], dX[2], L[7]);
	L[7] = fma(L[27], dX[2] * dX[0], L[7]);
	L[7] = fma(L[31], dX[2] * dX[1], L[7]);
	L[7] = fma(L[32], dX[2] * dX[2] * float(5.000000e-01), L[7]);
	L[6] = fma(L[12], dX[0], L[6]);
	L[6] = fma(L[22], dX[0] * dX[0] * float(5.000000e-01), L[6]);
	L[6] = fma(L[14], dX[1], L[6]);
	L[6] = fma(L[24], dX[1] * dX[0], L[6]);
	L[6] = fma(L[27], dX[1] * dX[1] * float(5.000000e-01), L[6]);
	L[6] = fma(L[15], dX[2], L[6]);
	L[6] = fma(L[25], dX[2] * dX[0], L[6]);
	L[6] = fma(L[28], dX[2] * dX[1], L[6]);
	L[6] = fma(L[29], dX[2] * dX[2] * float(5.000000e-01), L[6]);
	L[8] = fma(L[14], dX[0], L[8]);
	L[8] = fma(L[24], dX[0] * dX[0] * float(5.000000e-01), L[8]);
	L[8] = fma(L[17], dX[1], L[8]);
	L[8] = fma(L[27], dX[1] * dX[0], L[8]);
	L[8] = fma(L[31], dX[1] * dX[1] * float(5.000000e-01), L[8]);
	L[8] = fma(L[18], dX[2], L[8]);
	L[8] = fma(L[28], dX[2] * dX[0], L[8]);
	L[8] = fma(L[32], dX[2] * dX[1], L[8]);
	L[8] = fma(L[33], dX[2] * dX[2] * float(5.000000e-01), L[8]);
	L[9] = fma(L[15], dX[0], L[9]);
	L[9] = fma(L[25], dX[0] * dX[0] * float(5.000000e-01), L[9]);
	L[9] = fma(L[18], dX[1], L[9]);
	L[9] = fma(L[28], dX[1] * dX[0], L[9]);
	L[9] = fma(L[32], dX[1] * dX[1] * float(5.000000e-01), L[9]);
	L[9] = fma(L[19], dX[2], L[9]);
	L[9] = fma(L[29], dX[2] * dX[0], L[9]);
	L[9] = fma(L[33], dX[2] * dX[1], L[9]);
	L[9] = fma(L[34], dX[2] * dX[2] * float(5.000000e-01), L[9]);
	L[10] = fma(L[20], dX[0], L[10]);
	L[10] = fma(L[21], dX[1], L[10]);
	L[10] = fma(L[22], dX[2], L[10]);
	L[11] = fma(L[21], dX[0], L[11]);
	L[11] = fma(L[23], dX[1], L[11]);
	L[11] = fma(L[24], dX[2], L[11]);
	L[13] = fma(L[23], dX[0], L[13]);
	L[13] = fma(L[26], dX[1], L[13]);
	L[13] = fma(L[27], dX[2], L[13]);
	L[16] = fma(L[26], dX[0], L[16]);
	L[16] = fma(L[30], dX[1], L[16]);
	L[16] = fma(L[31], dX[2], L[16]);
	L[12] = fma(L[22], dX[0], L[12]);
	L[12] = fma(L[24], dX[1], L[12]);
	L[12] = fma(L[25], dX[2], L[12]);
	L[14] = fma(L[24], dX[0], L[14]);
	L[14] = fma(L[27], dX[1], L[14]);
	L[14] = fma(L[28], dX[2], L[14]);
	L[17] = fma(L[27], dX[0], L[17]);
	L[17] = fma(L[31], dX[1], L[17]);
	L[17] = fma(L[32], dX[2], L[17]);
	L[15] = fma(L[25], dX[0], L[15]);
	L[15] = fma(L[28], dX[1], L[15]);
	L[15] = fma(L[29], dX[2], L[15]);
	L[18] = fma(L[28], dX[0], L[18]);
	L[18] = fma(L[32], dX[1], L[18]);
	L[18] = fma(L[33], dX[2], L[18]);
	L[19] = fma(L[29], dX[0], L[19]);
	L[19] = fma(L[33], dX[1], L[19]);
	L[19] = fma(L[34], dX[2], L[19]);
	return L;
}

CUDA_EXPORT void shift_expansion(const expansion<float> &L, array<float, NDIM> &g, float &phi,
		const array<float, NDIM> &dX, bool do_phi) {
	float tmp1;

	if (do_phi) {
		float tmp2;
		phi = L[0];
		phi = fma(L[1], dX[0], phi);
		tmp1 = dX[0] * dX[0];
		phi = fma(L[4], tmp1 * float(5.000000e-01), phi);
		tmp2 = tmp1 * dX[0];
		phi = fma(L[10], tmp2 * float(1.666667e-01), phi);
		phi = fma(L[20], tmp2 * dX[0] * float(4.166667e-02), phi);
		phi = fma(L[2], dX[1], phi);
		tmp1 = dX[1] * dX[0];
		phi = fma(L[5], tmp1 * float(1.000000e+00), phi);
		tmp2 = tmp1 * dX[0];
		phi = fma(L[11], tmp2 * float(5.000000e-01), phi);
		phi = fma(L[21], tmp2 * dX[0] * float(1.666667e-01), phi);
		tmp1 = dX[1] * dX[1];
		phi = fma(L[7], tmp1 * float(5.000000e-01), phi);
		tmp2 = tmp1 * dX[0];
		phi = fma(L[13], tmp2 * float(5.000000e-01), phi);
		phi = fma(L[23], tmp2 * dX[0] * float(2.500000e-01), phi);
		tmp2 = tmp1 * dX[1];
		phi = fma(L[16], tmp2 * float(1.666667e-01), phi);
		phi = fma(L[26], tmp2 * dX[0] * float(1.666667e-01), phi);
		phi = fma(L[30], tmp2 * dX[1] * float(4.166667e-02), phi);
		phi = fma(L[3], dX[2], phi);
		tmp1 = dX[2] * dX[0];
		phi = fma(L[6], tmp1 * float(1.000000e+00), phi);
		tmp2 = tmp1 * dX[0];
		phi = fma(L[12], tmp2 * float(5.000000e-01), phi);
		phi = fma(L[22], tmp2 * dX[0] * float(1.666667e-01), phi);
		tmp1 = dX[2] * dX[1];
		phi = fma(L[8], tmp1 * float(1.000000e+00), phi);
		tmp2 = tmp1 * dX[0];
		phi = fma(L[14], tmp2 * float(1.000000e+00), phi);
		phi = fma(L[24], tmp2 * dX[0] * float(5.000000e-01), phi);
		tmp2 = tmp1 * dX[1];
		phi = fma(L[17], tmp2 * float(5.000000e-01), phi);
		phi = fma(L[27], tmp2 * dX[0] * float(5.000000e-01), phi);
		phi = fma(L[31], tmp2 * dX[1] * float(1.666667e-01), phi);
		tmp1 = dX[2] * dX[2];
		phi = fma(L[9], tmp1 * float(5.000000e-01), phi);
		tmp2 = tmp1 * dX[0];
		phi = fma(L[15], tmp2 * float(5.000000e-01), phi);
		phi = fma(L[25], tmp2 * dX[0] * float(2.500000e-01), phi);
		tmp2 = tmp1 * dX[1];
		phi = fma(L[18], tmp2 * float(5.000000e-01), phi);
		phi = fma(L[28], tmp2 * dX[0] * float(5.000000e-01), phi);
		phi = fma(L[32], tmp2 * dX[1] * float(2.500000e-01), phi);
		tmp2 = tmp1 * dX[2];
		phi = fma(L[19], tmp2 * float(1.666667e-01), phi);
		phi = fma(L[29], tmp2 * dX[0] * float(1.666667e-01), phi);
		phi = fma(L[33], tmp2 * dX[1] * float(1.666667e-01), phi);
		phi = fma(L[34], tmp2 * dX[2] * float(4.166667e-02), phi);
	}
	g[0] = -L[1];
	g[0] -= L[4] * dX[0];
	tmp1 = dX[0] * dX[0];
	g[0] = fma(L[10], tmp1 * float(-5.000000e-01), g[0]);
	g[0] = fma(L[20], tmp1 * dX[0] * float(-1.666667e-01), g[0]);
	g[0] -= L[5] * dX[1];
	tmp1 = dX[1] * dX[0];
	g[0] = fma(L[11], tmp1 * float(-1.000000e+00), g[0]);
	g[0] = fma(L[21], tmp1 * dX[0] * float(-5.000000e-01), g[0]);
	tmp1 = dX[1] * dX[1];
	g[0] = fma(L[13], tmp1 * float(-5.000000e-01), g[0]);
	g[0] = fma(L[23], tmp1 * dX[0] * float(-5.000000e-01), g[0]);
	g[0] = fma(L[26], tmp1 * dX[1] * float(-1.666667e-01), g[0]);
	g[0] -= L[6] * dX[2];
	tmp1 = dX[2] * dX[0];
	g[0] = fma(L[12], tmp1 * float(-1.000000e+00), g[0]);
	g[0] = fma(L[22], tmp1 * dX[0] * float(-5.000000e-01), g[0]);
	tmp1 = dX[2] * dX[1];
	g[0] = fma(L[14], tmp1 * float(-1.000000e+00), g[0]);
	g[0] = fma(L[24], tmp1 * dX[0] * float(-1.000000e+00), g[0]);
	g[0] = fma(L[27], tmp1 * dX[1] * float(-5.000000e-01), g[0]);
	tmp1 = dX[2] * dX[2];
	g[0] = fma(L[15], tmp1 * float(-5.000000e-01), g[0]);
	g[0] = fma(L[25], tmp1 * dX[0] * float(-5.000000e-01), g[0]);
	g[0] = fma(L[28], tmp1 * dX[1] * float(-5.000000e-01), g[0]);
	g[0] = fma(L[29], tmp1 * dX[2] * float(-1.666667e-01), g[0]);
	g[1] = -L[2];
	g[1] -= L[5] * dX[0];
	tmp1 = dX[0] * dX[0];
	g[1] = fma(L[11], tmp1 * float(-5.000000e-01), g[1]);
	g[1] = fma(L[21], tmp1 * dX[0] * float(-1.666667e-01), g[1]);
	g[1] -= L[7] * dX[1];
	tmp1 = dX[1] * dX[0];
	g[1] = fma(L[13], tmp1 * float(-1.000000e+00), g[1]);
	g[1] = fma(L[23], tmp1 * dX[0] * float(-5.000000e-01), g[1]);
	tmp1 = dX[1] * dX[1];
	g[1] = fma(L[16], tmp1 * float(-5.000000e-01), g[1]);
	g[1] = fma(L[26], tmp1 * dX[0] * float(-5.000000e-01), g[1]);
	g[1] = fma(L[30], tmp1 * dX[1] * float(-1.666667e-01), g[1]);
	g[1] -= L[8] * dX[2];
	tmp1 = dX[2] * dX[0];
	g[1] = fma(L[14], tmp1 * float(-1.000000e+00), g[1]);
	g[1] = fma(L[24], tmp1 * dX[0] * float(-5.000000e-01), g[1]);
	tmp1 = dX[2] * dX[1];
	g[1] = fma(L[17], tmp1 * float(-1.000000e+00), g[1]);
	g[1] = fma(L[27], tmp1 * dX[0] * float(-1.000000e+00), g[1]);
	g[1] = fma(L[31], tmp1 * dX[1] * float(-5.000000e-01), g[1]);
	tmp1 = dX[2] * dX[2];
	g[1] = fma(L[18], tmp1 * float(-5.000000e-01), g[1]);
	g[1] = fma(L[28], tmp1 * dX[0] * float(-5.000000e-01), g[1]);
	g[1] = fma(L[32], tmp1 * dX[1] * float(-5.000000e-01), g[1]);
	g[1] = fma(L[33], tmp1 * dX[2] * float(-1.666667e-01), g[1]);
	g[2] = -L[3];
	g[2] -= L[6] * dX[0];
	tmp1 = dX[0] * dX[0];
	g[2] = fma(L[12], tmp1 * float(-5.000000e-01), g[2]);
	g[2] = fma(L[22], tmp1 * dX[0] * float(-1.666667e-01), g[2]);
	g[2] -= L[8] * dX[1];
	tmp1 = dX[1] * dX[0];
	g[2] = fma(L[14], tmp1 * float(-1.000000e+00), g[2]);
	g[2] = fma(L[24], tmp1 * dX[0] * float(-5.000000e-01), g[2]);
	tmp1 = dX[1] * dX[1];
	g[2] = fma(L[17], tmp1 * float(-5.000000e-01), g[2]);
	g[2] = fma(L[27], tmp1 * dX[0] * float(-5.000000e-01), g[2]);
	g[2] = fma(L[31], tmp1 * dX[1] * float(-1.666667e-01), g[2]);
	g[2] -= L[9] * dX[2];
	tmp1 = dX[2] * dX[0];
	g[2] = fma(L[15], tmp1 * float(-1.000000e+00), g[2]);
	g[2] = fma(L[25], tmp1 * dX[0] * float(-5.000000e-01), g[2]);
	tmp1 = dX[2] * dX[1];
	g[2] = fma(L[18], tmp1 * float(-1.000000e+00), g[2]);
	g[2] = fma(L[28], tmp1 * dX[0] * float(-1.000000e+00), g[2]);
	g[2] = fma(L[32], tmp1 * dX[1] * float(-5.000000e-01), g[2]);
	tmp1 = dX[2] * dX[2];
	g[2] = fma(L[19], tmp1 * float(-5.000000e-01), g[2]);
	g[2] = fma(L[29], tmp1 * dX[0] * float(-5.000000e-01), g[2]);
	g[2] = fma(L[33], tmp1 * dX[1] * float(-5.000000e-01), g[2]);
	g[2] = fma(L[34], tmp1 * dX[2] * float(-1.666667e-01), g[2]);

}
