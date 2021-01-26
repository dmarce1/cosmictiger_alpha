/*
 * cuda.hpp
 *
 *  Created on: Jan 24, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_CUDA_HPP_
#define COSMICTIGER_CUDA_HPP_


#define CUDA_CHECK( a ) if( a != cudaSuccess ) printf( "CUDA error on line %i of %s : %s\n", __LINE__, __FILE__, cudaGetErrorString(a))

#define CUDA_EXPORT __device__ __host__
#define CUDA_KERNEL __global__ void

#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

template<class Archive>
void serialize(Archive &arc, cudaDeviceProp &props, unsigned int) {
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

struct cuda_properties {
   std::vector<cudaDeviceProp> devices;
   int num_devices;
   template<class A>
   void serialize(A&& arc, unsigned) {
      arc & devices;
      arc & num_devices;
   }
};

cuda_properties cuda_init();

#endif /* COSMICTIGER_CUDA_HPP_ */
