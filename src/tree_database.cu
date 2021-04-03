#define TREE_DATABASE_CU
#include <cosmictiger/tree.hpp>
#include <cosmictiger/tree_database.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/memory.hpp>

#include <cmath>
#include <atomic>

static constexpr int chunk_size = 2048;

static std::atomic<int> next_chunk;
static int era = 0;

static thread_local int current_alloc;
static thread_local int current_offset;
static thread_local int current_era = -1;

void tree_data_initialize() {
	gpu_tree_data_.ntrees = global().opts.nparts / global().opts.bucket_size;
	gpu_tree_data_.ntrees = std::max(1 << ((int) std::ceil(std::log(gpu_tree_data_.ntrees) / std::log(2)) + 3), 1024 * 1024);
	gpu_tree_data_.nchunks = gpu_tree_data_.ntrees / chunk_size;

	CUDA_MALLOC(gpu_tree_data_.radius, gpu_tree_data_.ntrees);
	for( int i = 0; i < NDIM; i++) {
		CUDA_MALLOC(gpu_tree_data_.pos[i], gpu_tree_data_.ntrees);
	}
	CUDA_MALLOC(gpu_tree_data_.leaf, gpu_tree_data_.ntrees);
	CUDA_MALLOC(gpu_tree_data_.multi, gpu_tree_data_.ntrees);
	CUDA_MALLOC(gpu_tree_data_.children, gpu_tree_data_.ntrees);
	CUDA_MALLOC(gpu_tree_data_.parts, gpu_tree_data_.ntrees);
	CUDA_MALLOC(gpu_tree_data_.active_parts, gpu_tree_data_.ntrees);
	CUDA_MALLOC(gpu_tree_data_.active_nodes, gpu_tree_data_.ntrees);

	for (int i = 0; i < gpu_tree_data_.ntrees; i++) {
		new (gpu_tree_data_.children + i) array<tree_ptr,NCHILD>();
	}

	printf("Allocating %i trees in %i chunks of %i each\n", gpu_tree_data_.ntrees, gpu_tree_data_.nchunks, chunk_size);

	tree_data_clear();

	cpu_tree_data_ = gpu_tree_data_;

}

void tree_data_clear() {
	next_chunk = 0;
	era++;
}

int tree_data_allocate() {
	if (era != current_era) {
		current_era = era;
		current_offset = chunk_size;
	}
	if (current_offset >= chunk_size) {
		current_alloc = chunk_size * next_chunk++;
		current_offset = 0;
	}
	if (current_alloc + chunk_size >= gpu_tree_data_.ntrees) {
		printf("Fatal error - tree arena full!\n");
		abort();
	}
	int alloc = current_offset++ + current_alloc;
	if (alloc == 0) { // reserve 0 for root
		alloc = tree_data_allocate();
	}
	//printf( "Allocating %i from chunk %i\n", alloc, alloc / chunk_size);
	return alloc;
}


