/*
 * kick_return.hpp
 *
 *  Created on: Mar 8, 2021
 *      Author: dmarce1
 */

#ifndef KICK_RETURN_HPP_
#define KICK_RETURN_HPP_

#include <cosmictiger/defs.hpp>
#include <cosmictiger/array.hpp>

#define KR_PP 0
#define KR_PC 1
#define KR_CP 2
#define KR_CC 3
#define KR_OP 4
#define KR_EWCC 5

#define KR_COUNT 6


struct kick_return {
	double phis;
	double kin;
	array<double,NDIM> mom;
	array<double,NDIM> forces;
	array<int,MAX_RUNG> rung_cnt;
	array<double,KR_COUNT> flop;
	array<double,KR_COUNT> count;
	int min_rung;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & phis;
		arc & kin;
		arc & mom;
		arc & forces;
		arc & rung_cnt;
		arc & flop;
		arc & count;
		arc & min_rung;
	}
};


kick_return kick_return_get();
void kick_return_update_pot_cpu( float, float, float, float);
__device__ void kick_return_update_pot_gpu( float, float, float, float);
void kick_return_update_rung_cpu(int rung);
__device__ void kick_return_update_rung_gpu(int rung);
__device__ void kick_return_update_interactions_gpu(int itype, int count, int flops);
void kick_return_update_interactions_cpu(int itype, int count, int flops);


void kick_return_init(int min_rung);

int kick_return_max_rung();
void kick_return_show();
int kick_return_pp_interactions();

#ifdef __CUDACC__
#ifndef KICK_RETURN_CU
extern __managed__ kick_return gpu_return;
#else
__managed__ kick_return gpu_return;
#endif

inline __device__ void kick_return_update_interactions_gpu(int itype, int count, int flops) {
	for (int P = KICK_BLOCK_SIZE / 2; P >= 1; P /= 2) {
		flops += __shfl_down_sync(0xFFFFFFFF, flops, P);
		count += __shfl_down_sync(0xFFFFFFFF, count, P);
	}
	if (threadIdx.x == 0) {
		atomicAdd(&gpu_return.flop[itype], flops);
		atomicAdd(&gpu_return.count[itype], count);
	}
}

inline __device__ void kick_return_update_rung_gpu(int rung) {
	atomicAdd(&gpu_return.rung_cnt[rung], 1);
}

inline __device__ void kick_return_update_pot_gpu(float phi, float fx, float fy, float fz) {
	for (int P = warpSize / 2; P >= 1; P /= 2) {
		phi += __shfl_xor_sync(FULL_MASK, phi, P);
		fx += __shfl_xor_sync(FULL_MASK, fx, P);
		fy += __shfl_xor_sync(FULL_MASK, fy, P);
		fz += __shfl_xor_sync(FULL_MASK, fz, P);
	}
	if (threadIdx.x == 0) {
		atomicAdd(&gpu_return.phis, phi);
		atomicAdd(&gpu_return.forces[0], fx);
		atomicAdd(&gpu_return.forces[1], fy);
		atomicAdd(&gpu_return.forces[2], fz);
	}

}

#endif

#endif /* KICK_RETURN_HPP_ */
