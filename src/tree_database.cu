#define TREE_DATABASE_CU
#include <cosmictiger/tree.hpp>
#include <cosmictiger/tree_database.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/memory.hpp>

#include <cmath>
#include <atomic>

static const int min_trees = 1024 * 1024;

static std::atomic<int> next_chunk;

int hardware_concurrency();

template<class T>
void free_if_needed(T** ptr) {
	if (*ptr) {
		CUDA_FREE(*ptr);
		*ptr = nullptr;
	}
}

void tree_data_initialize_kick() {

	gpu_tree_data_.chunk_size = 1;
	gpu_tree_data_.ntrees = 5 * global().opts.nparts / global().opts.bucket_size;
	if (global().opts.sph) {
		gpu_tree_data_.ntrees *= 2;
	}
	gpu_tree_data_.ntrees = std::max(gpu_tree_data_.ntrees, min_trees);
	const int target_chunk_size = gpu_tree_data_.ntrees / (16 * OVERSUBSCRIPTION * hardware_concurrency());
	while (gpu_tree_data_.chunk_size < target_chunk_size) {
		gpu_tree_data_.chunk_size *= 2;
	}
	gpu_tree_data_.nchunks = gpu_tree_data_.ntrees / gpu_tree_data_.chunk_size;

//	CUDA_CHECK(cudaMemAdvise(&gpu_tree_data_, sizeof(gpu_tree_data_), cudaMemAdviseSetReadMostly, 0));

	printf("Allocating %i trees in %i chunks of %i each for kick\n", gpu_tree_data_.ntrees, gpu_tree_data_.nchunks,
			gpu_tree_data_.chunk_size);

	tree_data_free_all();
	CUDA_MALLOC(gpu_tree_data_.data, gpu_tree_data_.ntrees);
	CUDA_MALLOC(gpu_tree_data_.parts, gpu_tree_data_.ntrees);
	CUDA_MALLOC(gpu_tree_data_.multi, gpu_tree_data_.ntrees);
	CUDA_MALLOC(gpu_tree_data_.active_nodes, gpu_tree_data_.ntrees);
	CUDA_MALLOC(gpu_tree_data_.active_parts, gpu_tree_data_.ntrees);
	CUDA_MALLOC(gpu_tree_data_.all_local, gpu_tree_data_.ntrees);
	if (global().opts.sph) {
		CUDA_MALLOC(gpu_tree_data_.ranges, gpu_tree_data_.ntrees);
		CUDA_MALLOC(gpu_tree_data_.sph_ranges, gpu_tree_data_.ntrees);
	}
	tree_data_clear();

	cpu_tree_data_ = gpu_tree_data_;

}

void tree_data_initialize_groups() {
	gpu_tree_data_.chunk_size = 1;
	gpu_tree_data_.ntrees = 4 * global().opts.nparts / GROUP_BUCKET_SIZE;
	gpu_tree_data_.ntrees = std::max(gpu_tree_data_.ntrees, min_trees);
	const int target_chunk_size = gpu_tree_data_.ntrees / (16 * OVERSUBSCRIPTION * hardware_concurrency());
	while (gpu_tree_data_.chunk_size < target_chunk_size) {
		gpu_tree_data_.chunk_size *= 2;
	}
	gpu_tree_data_.nchunks = gpu_tree_data_.ntrees / gpu_tree_data_.chunk_size;

//	CUDA_CHECK(cudaMemAdvise(&gpu_tree_data_, sizeof(gpu_tree_data_), cudaMemAdviseSetReadMostly, 0));

	printf("Allocating %i trees in %i chunks of %i each for group search\n", gpu_tree_data_.ntrees,
			gpu_tree_data_.nchunks, gpu_tree_data_.chunk_size);

	tree_data_free_all();
	CUDA_MALLOC(gpu_tree_data_.data, gpu_tree_data_.ntrees);
	CUDA_MALLOC(gpu_tree_data_.parts, gpu_tree_data_.ntrees);
	CUDA_MALLOC(gpu_tree_data_.ranges, gpu_tree_data_.ntrees);
	CUDA_MALLOC(gpu_tree_data_.active_parts, gpu_tree_data_.ntrees);
	CUDA_MALLOC(gpu_tree_data_.active_nodes, gpu_tree_data_.ntrees);
	CUDA_MALLOC(gpu_tree_data_.all_local, gpu_tree_data_.ntrees);

	tree_data_clear();

	cpu_tree_data_ = gpu_tree_data_;

}

size_t tree_data_bytes_used() {
	size_t use = 0;
	if (gpu_tree_data_.data) {
		use += sizeof(tree_data_t);
	}
	if (gpu_tree_data_.parts) {
		use += sizeof(pair<size_t, size_t> );
	}
	if (gpu_tree_data_.multi) {
		use += sizeof(multipole_pos);
	}
	if (gpu_tree_data_.ranges) {
		use += sizeof(range);
	}
	if (gpu_tree_data_.active_nodes) {
		use += sizeof(size_t);
	}
	if (gpu_tree_data_.active_parts) {
		use += sizeof(size_t);
	}
	if (gpu_tree_data_.all_local) {
		use += sizeof(bool);
	}
	use *= gpu_tree_data_.nchunks * gpu_tree_data_.chunk_size;
	return use;
}


void tree_data_free_all_cu() {
	free_if_needed(&gpu_tree_data_.data);
	free_if_needed(&gpu_tree_data_.parts);
	free_if_needed(&gpu_tree_data_.multi);
	free_if_needed(&gpu_tree_data_.sph_ranges);
	free_if_needed(&gpu_tree_data_.ranges);
	free_if_needed(&gpu_tree_data_.active_nodes);
	free_if_needed(&gpu_tree_data_.active_parts);
	free_if_needed(&gpu_tree_data_.all_local);
}

void tree_data_set_groups_cu() {
	for (int i = 0; i < gpu_tree_data_.ntrees; i++) {
		gpu_tree_data_.active_parts[i] = gpu_tree_data_.active_nodes[i];
		gpu_tree_data_.active_nodes[i] = 0;
	}
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
	tree_cache_clear();
	next_chunk = 0;
	if (gpu_tree_data_.data) {
		for (int i = 0; i < gpu_tree_data_.ntrees; i++) {
			gpu_tree_data_.data[i].children[0].dindex = -1;
		}
	}
}

std::pair<int, int> tree_data_allocate() {
	std::pair<int, int> rc;
	const int chunk = next_chunk++;
	if (chunk >= gpu_tree_data_.nchunks) {
		printf("Fatal error - tree arena full!\n");
		abort();
	}
	rc.first = chunk * gpu_tree_data_.chunk_size;
	rc.second = rc.first + gpu_tree_data_.chunk_size;
	return rc;
}

double tree_data_use() {
	return (double) next_chunk / (double) gpu_tree_data_.nchunks;
}

