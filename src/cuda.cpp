/*
 * cuda.cpp
 *
 *  Created on: Jan 25, 2021
 *      Author: dmarce1
 */

#include <cosmictiger/cuda.hpp>

#include <cosmictiger/hpx.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/drift.hpp>
#include <cstdlib>

CUDA_KERNEL cuda_kick_kernel(kick_params_type *params);
CUDA_KERNEL cuda_pp_ewald_interactions(particle_set *parts, size_t *test_parts, float *ferr, float *fnorm, float* perr,
		float* pnorm, float GM, float);

#define STACK_SIZE (32*1024)
#define HEAP_SIZE (size_t(2)*1024*1024*1024)
#define RECUR_LIMIT 8
#define L2FETCH 64
#define PENDINGLAUNCHES 128

HPX_PLAIN_ACTION (cuda_init);

cuda_properties cuda_init() {
	hpx::future<cuda_properties> futl, futr;
	auto children = hpx_child_localities();
	if (children.first != hpx::invalid_id) {
		futl = hpx::async<cuda_init_action>(children.first);
	}
	if (children.second != hpx::invalid_id) {
		futr = hpx::async<cuda_init_action>(children.second);
	}

	cuda_properties props;
	CUDA_CHECK(cudaGetDeviceCount(&props.num_devices));
	props.devices.resize(props.num_devices);
	for (int i = 0; i < props.num_devices; i++) {
		CUDA_CHECK(cudaGetDeviceProperties(&props.devices[i], i));
	}
	printf("--------------------------------------------------------------------------------\n");
	printf("Detected %i CUDA devices.\n", props.num_devices);
	for (int i = 0; i < props.num_devices; i++) {
	}
	CUDA_CHECK(cudaDeviceReset());
	size_t value = STACK_SIZE;
	CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, value));
	CUDA_CHECK(cudaDeviceGetLimit(&value, cudaLimitStackSize));
	bool fail = false;
	if (value != STACK_SIZE) {
		printf("Unable to set stack size to %i\n", STACK_SIZE);
		fail = true;
	}
	value = HEAP_SIZE;
	CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, value));
	CUDA_CHECK(cudaDeviceGetLimit(&value, cudaLimitMallocHeapSize));
	if (value != HEAP_SIZE) {
		printf("Unable to set heap to %li\n", HEAP_SIZE);
		fail = true;
	}
	value = RECUR_LIMIT;
	CUDA_CHECK(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, value));
	CUDA_CHECK(cudaDeviceGetLimit(&value, cudaLimitDevRuntimeSyncDepth));
	if (value != RECUR_LIMIT) {
		printf("Unable to set recursion limit to %i\n", RECUR_LIMIT);
		fail = true;
	}
	value = L2FETCH;
	CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, value));
	CUDA_CHECK(cudaDeviceGetLimit(&value, cudaLimitMaxL2FetchGranularity));
	if (value != L2FETCH) {
		printf("Unable to set L2 fetch granularity to to %i\n", L2FETCH);
		fail = true;
	}

	if (fail) {
		abort();
	}

	if (futl.valid()) {
		futl.get();
	}
	if (futr.valid()) {
		futr.get();
	}
	return props;
}

void execute_host_function(void *data) {
	auto* fptr = (std::function<void()>*) data;
	(*fptr)();
	delete fptr;
}

void cuda_enqueue_host_function(cudaStream_t stream, std::function<void()>&& function) {
	auto* fptr = new std::function<void()>(std::move(function));
	CUDA_CHECK(cudaLaunchHostFunc(stream, &execute_host_function, fptr));
}

