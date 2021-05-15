#include <cosmictiger/particle.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/global.hpp>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
void particle_set::generate_random(int seed) {

	if (size_) {
		cudaFuncAttributes attribs;
		CUDA_CHECK(cudaFuncGetAttributes(&attribs, generate_random_vectors));
		int num_threads = attribs.maxThreadsPerBlock;
		int num_blocks;
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, generate_random_vectors, num_threads, 0));
		num_blocks *= global().cuda.devices[0].multiProcessorCount;
		generate_random_vectors<<<num_blocks,num_threads>>>(xptr_[0]+offset_,xptr_[1]+offset_,xptr_[2]+offset_,size_,seed);
		CUDA_CHECK(cudaDeviceSynchronize());

		for (int i = offset_; i < size_ + offset_; i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				vel(0, i) = 0.f;
				vel(1, i) = 0.f;
				vel(2, i) = 0.f;
			}
			set_rung(0, i);
		}
	}
}

CUDA_KERNEL generate_morton_keys(fixed32* x, fixed32* y, fixed32* z, morton_t* keys, size_t size) {
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	const int bsz = blockDim.x;
	const int gsz = gridDim.x;
	const size_t start = (bid) * size / gsz;
	const size_t stop = (bid + 1) * size / gsz;
	for (size_t i = start + tid; i < stop; i += bsz) {
		keys[i] = morton_key(x[i], y[i], z[i]);
	}
}

void particle_set::generate_keys() {
	cudaFuncAttributes attribs;
	CUDA_CHECK(cudaFuncGetAttributes(&attribs, generate_morton_keys));
	int num_threads = attribs.maxThreadsPerBlock;
	int num_blocks;
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, generate_morton_keys, num_threads, 0));
	num_blocks *= global().cuda.devices[0].multiProcessorCount;
	generate_morton_keys<<<num_blocks,num_threads>>>(xptr_[0]+offset_,xptr_[1]+offset_,xptr_[2]+offset_, keyptr_+offset_,size_);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void particle_set::sort_keys() {
	thrust::sort(thrust::device, keyptr_, keyptr_ + size_);
}
