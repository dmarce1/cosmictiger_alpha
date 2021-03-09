

#include <cosmictiger/kick_return.hpp>
#include <cuda_runtime.h>


static __managed__ kick_return gpu_return;


__global__ void kick_return_init_kernel(int min_rung) {
	gpu_return.min_rung = min_rung;
	for( int i = 0; i < MAX_RUNG; i++) {
		gpu_return.rung_cnt[i] = 0;
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
