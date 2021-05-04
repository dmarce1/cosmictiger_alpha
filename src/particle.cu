#include <cosmictiger/particle.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/global.hpp>

void particle_set::generate_random(int seed) {

	if (size_) {
		cudaFuncAttributes attribs;
		CUDA_CHECK(cudaFuncGetAttributes(&attribs, generate_random_vectors));
		int num_threads = attribs.maxThreadsPerBlock;
		int num_blocks;
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, generate_random_vectors, num_threads, 0));
		num_blocks *= global().cuda.devices[0].multiProcessorCount;
		printf( "%i x %i\n", num_blocks,num_threads);
		generate_random_vectors<<<num_blocks,num_threads>>>(xptr_[0],xptr_[1],xptr_[2],global().opts.nparts,seed);
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
