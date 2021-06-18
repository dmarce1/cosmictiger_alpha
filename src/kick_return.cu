#define KICK_RETURN_CU

#include <cosmictiger/kick_return.hpp>
#include <cuda_runtime.h>


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



void kick_return_init_gpu(int min_rung) {
	kick_return_init_kernel<<<1,1>>>(min_rung);
	CUDA_CHECK(cudaDeviceSynchronize());
}

kick_return kick_return_get_gpu() {
	return gpu_return;
}
