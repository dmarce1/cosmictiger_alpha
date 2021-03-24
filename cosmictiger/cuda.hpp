/*
 * cuda.hpp
 *
 *  Created on: Jan 24, 2021
 *      Author: dmarce1
 */


#ifndef COSMICTIGER_CUDA_HPP_
#define COSMICTIGER_CUDA_HPP_


#include <functional>

#define CUDA_CHECK( a ) if( a != cudaSuccess ) printf( "CUDA error on line %i of %s : %s\n", __LINE__, __FILE__, cudaGetErrorString(a))

//#define CUDA_CHECK( a ) printf( "calling %s from %s line %i\n", #a, __FILE__, __LINE__ ); \
//		                  if( a != cudaSuccess ) printf( "CUDA error on line %i of %s : %s\n", __LINE__, __FILE__, cudaGetErrorString(a)); \
//		                  printf( "done calling %s from %s line %i\n", #a, __FILE__, __LINE__)

#ifdef __CUDACC__
#define CUDA_EXPORT __device__ __host__
#else
#define CUDA_EXPORT
#endif

#define CUDA_DEVICE __device__
#define CUDA_KERNEL __global__ void

#ifdef __CUDA_ARCH__
#define CUDA_SYNC() __threadfence_block()
#else
#define CUDA_SYNC()
#endif

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

void cuda_enqueue_host_function(cudaStream_t stream, std::function<void()>&& function);


#ifdef __CUDACC__
__device__ inline void cuda_sync() {
	__threadfence_block();
}
#endif


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

template<class T>
inline CUDA_EXPORT void nan_test(T a, const char* file, int line ) {
		if( isnan(a)) {
			printf( "NaN found %s %i\n", file, line);
	#ifdef __CUDA_ARCH__
			asm("trap;");
	#else
			abort();
	#endif

		}
		if( isinf(a)) {
			printf( "inf found %s %i\n", file, line);
	#ifdef __CUDA_ARCH__
			asm("trap;");
	#else
			abort();
	#endif

		}
}


#ifdef USE_NAN_TEST
#define NAN_TEST(a) nan_test(a,__FILE__,__LINE__)
#else
#define NAN_TEST(a)
#endif


#endif /* COSMICTIGER_CUDA_HPP_ */
