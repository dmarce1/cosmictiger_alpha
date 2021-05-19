#include <cosmictiger/kick_return.hpp>
#include <cuda_runtime.h>

static __managed__ kick_return gpu_return;

__global__ void kick_return_init_kernel(int min_rung) {
	gpu_return.min_rung = min_rung;
	for (int i = 0; i < MAX_RUNG; i++) {
		gpu_return.rung_cnt[i] = 0;
	}
	for (int i = 0; i < KR_COUNT; i++) {
		gpu_return.flop[i] = 0;
		gpu_return.count[i] = 0;
	}
	gpu_return.phis = 0.f;
	for (int dim = 0; dim < NDIM; dim++) {
		gpu_return.forces[dim] = 0.f;
	}
}

__device__ void kick_return_update_interactions_gpu(int itype, int count, int flops) {
	for (int P = KICK_BLOCK_SIZE / 2; P >= 1; P /= 2) {
		flops += __shfl_down_sync(0xFFFFFFFF, flops, P);
		count += __shfl_down_sync(0xFFFFFFFF, count, P);
	}
	if (threadIdx.x == 0) {
		atomicAdd(&gpu_return.flop[itype], flops);
		atomicAdd(&gpu_return.count[itype], count);
	}
}

void kick_return_init_gpu(int min_rung) {
	kick_return_init_kernel<<<1,1>>>(min_rung);
	CUDA_CHECK(cudaDeviceSynchronize());
}

kick_return kick_return_get_gpu() {
	return gpu_return;
}

__device__ void kick_return_update_rung_gpu(int rung) {
	atomicAdd(&gpu_return.rung_cnt[rung], 1);
}

__device__ void kick_return_update_pot_gpu(float phi, float fx, float fy, float fz) {
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
