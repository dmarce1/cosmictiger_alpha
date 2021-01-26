/*
 * cuda.hpp
 *
 *  Created on: Jan 24, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_CUDA_HPP_
#define COSMICTIGER_CUDA_HPP_

#define CUDA_CHECK( a ) if( a != cudaSuccess ) printf( "CUDA error on line %i of %s : %s\n", __LINE__, __FILE__, cudaGetErrorString(a))
#include <cuda_runtime.h>

template<class Archive>
void serialize(Archive &A, cudaDeviceProp &props, unsigned int) {
   for (int i = 0; i < 256; i++) {
      arc & props.name[i];
   }
   arc & props.totalGlobalMem;
   arc & props.sharedMemPerBlock;
   arc & props.regsPerBlock;
   arc & props.warpSize;
   arc & props.memPitch;
   arc & props.maxThreadsPerBlock;
   arc & props.maxThreadsDim[3];
   arc & props.maxGridSize[3];
   arc & props.totalConstMem;
   arc & props.major;
   arc & props.minor;
   arc & props.clockRate;
   arc & props.textureAlignment;
   arc & props.deviceOverlap;
   arc & props.multiProcessorCount;
   arc & props.kernelExecTimeoutEnabled;
   arc & props.integrated;
   arc & props.canMapHostMemory;
   arc & props.computeMode;
   arc & props.concurrentKernels;
   arc & props.ECCEnabled;
   arc & props.pciBusID;
   arc & props.pciDeviceID;
   arc & props.tccDriver;
}

cudaDeviceProp cuda_init();

#endif /* COSMICTIGER_CUDA_HPP_ */
