/*
 * cuda.cpp
 *
 *  Created on: Jan 25, 2021
 *      Author: dmarce1
 */

#include <cosmictiger/cuda.hpp>
#include <cstdlib>

#define STACK_SIZE (4*1024)
#define HEAP_SIZE 4*1024*1024
#define RECUR_LIMIT 0

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
   value = RECUR_LIMIT;
   CUDA_CHECK(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, value));
   CUDA_CHECK(cudaDeviceGetLimit(&value, cudaLimitDevRuntimeSyncDepth));
   if (value != RECUR_LIMIT) {
      printf("Unable to set recursion limit to %i\n", RECUR_LIMIT);
      fail = true;
   }
   if (fail) {
      abort();
   }

   return props;
}

