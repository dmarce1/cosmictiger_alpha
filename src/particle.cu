#include <cosmictiger/particle.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/global.hpp>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>

CUDA_EXPORT bool operator<(const particle& a, const particle& b) {
	for (int dim = 0; dim < NDIM; dim++) {
		if (a.x[dim].to_double() < b.x[dim].to_double()) {
			return true;
		} else if (a.x[dim].to_double() > b.x[dim].to_double()) {
			return false;
		}
	}
	return false;
}

void particle_set::sort_parts(particle* begin, particle* end) {
	thrust::sort(thrust::device, begin, end);
}

void particle_set::sort_indices(part_int* begin, part_int* end) {
	thrust::sort(thrust::device, begin, end);
}

void particle_set::generate_random(int seed) {

	if (size_) {
		cudaFuncAttributes attribs;
		CUDA_CHECK(cudaFuncGetAttributes(&attribs, generate_random_vectors));
		int num_threads = attribs.maxThreadsPerBlock;
		int num_blocks;
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, generate_random_vectors, num_threads, 0));
		num_blocks *= global().cuda.devices[0].multiProcessorCount;
		generate_random_vectors<<<num_blocks,num_threads>>>(xptr_[0],xptr_[1],xptr_[2],size_,seed);
		CUDA_CHECK(cudaDeviceSynchronize());

		for (int i = 0; i < size_; i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				vel(0, i) = 0.f;
				vel(1, i) = 0.f;
				vel(2, i) = 0.f;
			}
			set_rung(0, i);
		}
	}
}

