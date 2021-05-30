/*
 * kernel.hpp
 *
 *  Created on: May 30, 2021
 *      Author: dmarce1
 */

#ifndef KERNEL_HPP_
#define KERNEL_HPP_

#include <cosmictiger/global.hpp>


template<class F, class ...Args>
void execute_kernel(F&& kernel, Args&& ... args) {
	cuda_set_device();
	cudaFuncAttributes attribs;
	CUDA_CHECK(cudaFuncGetAttributes(&attribs, kernel));
	int num_threads = attribs.maxThreadsPerBlock;
	int num_blocks;
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, kernel, num_threads, 0));
	num_blocks *= global().cuda.devices[0].multiProcessorCount;
	kernel<<<num_blocks,num_threads>>>(std::forward<Args>(args)...);
	CUDA_CHECK(cudaDeviceSynchronize());
}




#endif /* KERNEL_HPP_ */
