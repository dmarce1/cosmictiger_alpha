#define TREE_DATABASE_CU
#include <cosmictiger/tree.hpp>
#include <cosmictiger/tree_database.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/memory.hpp>

#include <cmath>
#include <atomic>

static constexpr int chunk_size = 2048;

static std::atomic<int> next_chunk;

void tree_data_initialize() {
	gpu_tree_data_.ntrees = global().opts.nparts / global().opts.bucket_size;
	gpu_tree_data_.ntrees = std::max(1 << ((int) std::ceil(std::log(gpu_tree_data_.ntrees) / std::log(2)) + 3),
			2*1024 * 1024);
	gpu_tree_data_.nchunks = gpu_tree_data_.ntrees / chunk_size;

	CUDA_CHECK(cudaMemAdvise(&gpu_tree_data_, sizeof(gpu_tree_data_), cudaMemAdviseSetReadMostly, 0));

	printf("Allocating %i trees in %i chunks of %i each\n", gpu_tree_data_.ntrees, gpu_tree_data_.nchunks, chunk_size);

	CUDA_MALLOC(gpu_tree_data_.data, gpu_tree_data_.ntrees);
	CUDA_MALLOC(gpu_tree_data_.parts, gpu_tree_data_.ntrees);
	CUDA_MALLOC(gpu_tree_data_.multi, gpu_tree_data_.ntrees);
	CUDA_MALLOC(gpu_tree_data_.ranges, gpu_tree_data_.ntrees);
	CUDA_MALLOC(gpu_tree_data_.active_nodes, gpu_tree_data_.ntrees);
	CUDA_MALLOC(gpu_tree_data_.active_parts, gpu_tree_data_.ntrees);


	tree_data_clear();

	cpu_tree_data_ = gpu_tree_data_;

}


void tree_database_set_readonly() {
#ifdef USE_READMOSTLY
	CUDA_CHECK(cudaMemAdvise(gpu_tree_data_.data, gpu_tree_data_.ntrees, cudaMemAdviseSetReadMostly, 0));
	CUDA_CHECK(cudaMemAdvise(gpu_tree_data_.active_nodes, gpu_tree_data_.ntrees, cudaMemAdviseSetReadMostly, 0));
#endif
}

void tree_database_unset_readonly() {
#ifdef USE_READMOSTLY
	CUDA_CHECK(cudaMemAdvise(gpu_tree_data_.data, gpu_tree_data_.ntrees, cudaMemAdviseUnsetReadMostly, 0));
	CUDA_CHECK(cudaMemAdvise(gpu_tree_data_.active_nodes, gpu_tree_data_.ntrees, cudaMemAdviseUnsetReadMostly, 0));
#endif
}


void tree_data_clear() {
	next_chunk = 0;
	for (int i = 0; i < gpu_tree_data_.ntrees; i++) {
		gpu_tree_data_.data[i].children[0].dindex = -1;
	}
}

std::pair<int,int> tree_data_allocate() {
	std::pair<int,int> rc;
	const int chunk = next_chunk++;
	if( chunk >= gpu_tree_data_.nchunks) {
		printf("Fatal error - tree arena full!\n");
		abort();
	}
	rc.first = chunk * chunk_size;
	rc.second = rc.first + chunk_size;
	return rc;
}

