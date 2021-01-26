/*
 * cuda.cpp
 *
 *  Created on: Jan 25, 2021
 *      Author: dmarce1
 */

#include <cosmictiger/cuda.hpp>

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
   return props;
}
