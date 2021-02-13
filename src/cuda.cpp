/*
 * cuda.cpp
 *
 *  Created on: Jan 25, 2021
 *      Author: dmarce1
 */

#include <cosmictiger/cuda.hpp>
#include <cstdlib>

#define STACK_SIZE (32*1024)
#define HEAP_SIZE 512*1024*1024
#define RECUR_LIMIT 0
#define L2FETCH 32
#define PENDINGLAUNCHES 128

cuda_properties cuda_init() {
   cuda_properties props;
   CUDA_CHECK(cudaGetDeviceCount(&props.num_devices));
   props.devices.resize(props.num_devices);
   for (int i = 0; i < props.num_devices; i++) {
      CUDA_CHECK(cudaGetDeviceProperties(&props.devices[i], i));
   }
   printf("--------------------------------------------------------------------------------\n");
   printf("Detected %i CUDA devices.\n", props.num_devices);
   printf("Multi-processor counts: \n");
   for (int i = 0; i < props.num_devices; i++) {
      printf("\t %s: %i\n", props.devices[i].name, props.devices[i].multiProcessorCount);
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
      printf("Unable to set heap to %i\n", HEAP_SIZE);
      fail = true;
   }
   return props;
   value = RECUR_LIMIT;
   CUDA_CHECK(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, value));
   CUDA_CHECK(cudaDeviceGetLimit(&value, cudaLimitDevRuntimeSyncDepth));
   if (value != RECUR_LIMIT) {
      printf("Unable to set recursion limit to %i\n", RECUR_LIMIT);
      fail = true;
   }
   value = L2FETCH;
   CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity , value));
   CUDA_CHECK(cudaDeviceGetLimit(&value, cudaLimitMaxL2FetchGranularity ));
   if (value != L2FETCH) {
      printf("Unable to set L2 fetch granularity to to %li\n", L2FETCH);
      fail = true;
   }
//   value = PENDINGLAUNCHES;
//   CUDA_CHECK(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount , value));
//   CUDA_CHECK(cudaDeviceGetLimit(&value, cudaLimitDevRuntimePendingLaunchCount ));
//   if (value != L2FETCH) {
//      printf("Unable to set pending launch count to %li\n",  PENDINGLAUNCHES);
//      fail = true;
//   }
   CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
   //CUDA_CHECK(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeDefault));
 //   CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
   if (fail) {
      abort();
   }

   return props;
}

