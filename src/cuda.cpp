/*
 * cuda.cpp
 *
 *  Created on: Jan 25, 2021
 *      Author: dmarce1
 */

#include <cosmictiger/cuda.hpp>
#include <cstdlib>

#define STACK_SIZE (4*1024)

cuda_properties cuda_init() {
   cuda_properties props;
   CUDA_CHECK(cudaGetDeviceCount(&props.num_devices));
   props.devices.resize(props.num_devices);
   for (int i = 0; i < props.num_devices; i++) {
      CUDA_CHECK(cudaGetDeviceProperties(&props.devices[i], i));
   }
   printf( "--------------------------------------------------------------------------------\n");
   printf( "Detected %i CUDA devices.\n", props.num_devices);
   printf( "Multi-processor counts: \n");
   for( int i = 0; i < props.num_devices; i++) {
      printf( "\t %s: %i\n", props.devices[i].name, props.devices[i].multiProcessorCount);
   }
   size_t stack_size = STACK_SIZE;
   CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, stack_size));
   CUDA_CHECK(cudaDeviceGetLimit(&stack_size, cudaLimitStackSize));
   if( stack_size != STACK_SIZE) {
      printf( "Unable to set CUDA stack to %i bytes\n", STACK_SIZE);
      abort();
   }

   return props;
}

