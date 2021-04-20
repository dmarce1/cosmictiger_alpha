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
	gpu_tree_data_.ntrees = 4 * global().opts.nparts / global().opts.bucket_size;
	gpu_tree_data_.ntrees = std::max(gpu_tree_data_.ntrees, min_trees);
	const int target_chunk_size = gpu_tree_data_.ntrees / (16 * OVERSUBSCRIPTION * hardware_concurrency());
	while (gpu_tree_data_.chunk_size < target_chunk_size) {
		gpu_tree_data_.chunk_size *= 2;
	}
	gpu_tree_data_.nchunks = gpu_tree_data_.ntrees / gpu_tree_data_.chunk_size;

//	CUDA_CHECK(cudaMemAdvise(&gpu_tree_data_, sizeof(gpu_tree_data_), cudaMemAdviseSetReadMostly, 0));

	printf("Allocating %i trees in %i chunks of %i each for kick\n", gpu_tree_data_.ntrees, gpu_tree_data_.nchunks,
			gpu_tree_data_.chunk_size);

	free_if_needed(&gpu_tree_data_.data);
	free_if_needed(&gpu_tree_data_.parts);
	free_if_needed(&gpu_tree_data_.multi);
	free_if_needed(&gpu_tree_data_.ranges);
	free_if_needed(&gpu_tree_data_.active_nodes);
	free_if_needed(&gpu_tree_data_.active_parts);
	free_if_needed(&gpu_tree_data_.group_flags);
	free_if_needed(&gpu_tree_data_.last_group_flags);
	CUDA_MALLOC(gpu_tree_data_.data, gpu_tree_data_.ntrees);
	CUDA_MALLOC(gpu_tree_data_.parts, gpu_tree_data_.ntrees);
	CUDA_MALLOC(gpu_tree_data_.multi, gpu_tree_data_.ntrees);
	CUDA_MALLOC(gpu_tree_data_.active_nodes, gpu_tree_data_.ntrees);
	CUDA_MALLOC(gpu_tree_data_.active_parts, gpu_tree_data_.ntrees);
	CUDA_MALLOC(gpu_tree_data_.group_flags, gpu_tree_data_.ntrees);
	CUDA_MALLOC(gpu_tree_data_.last_group_flags, gpu_tree_data_.ntrees);

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

	free_if_needed(&gpu_tree_data_.data);
	free_if_needed(&gpu_tree_data_.parts);
	free_if_needed(&gpu_tree_data_.multi);
	free_if_needed(&gpu_tree_data_.ranges);
	free_if_needed(&gpu_tree_data_.active_nodes);
	free_if_needed(&gpu_tree_data_.active_parts);
	free_if_needed(&gpu_tree_data_.group_flags);
	free_if_needed(&gpu_tree_data_.last_group_flags);
	CUDA_MALLOC(gpu_tree_data_.data, gpu_tree_data_.ntrees);
	CUDA_MALLOC(gpu_tree_data_.parts, gpu_tree_data_.ntrees);
	CUDA_MALLOC(gpu_tree_data_.ranges, gpu_tree_data_.ntrees);
	CUDA_MALLOC(gpu_tree_data_.group_flags, gpu_tree_data_.ntrees);
	CUDA_MALLOC(gpu_tree_data_.last_group_flags, gpu_tree_data_.ntrees);

	tree_data_clear();

	cpu_tree_data_ = gpu_tree_data_;

}

void tree_data_free_all() {
	free_if_needed(&gpu_tree_data_.data);
	free_if_needed(&gpu_tree_data_.parts);
	free_if_needed(&gpu_tree_data_.multi);
	free_if_needed(&gpu_tree_data_.ranges);
	free_if_needed(&gpu_tree_data_.active_nodes);
	free_if_needed(&gpu_tree_data_.active_parts);
	free_if_needed(&gpu_tree_data_.group_flags);
	free_if_needed(&gpu_tree_data_.last_group_flags);
}

void tree_database_reset_group_flags() {
	for (int i = 0; i < gpu_tree_data_.ntrees; i++) {
		gpu_tree_data_.group_flags[i] = 1;
	}
	for (int i = 0; i < gpu_tree_data_.ntrees; i++) {
		gpu_tree_data_.last_group_flags[i] = 1;
	}
}

void tree_database_set_last_group_flags() {
	for (int i = 0; i < gpu_tree_data_.ntrees; i++) {
		gpu_tree_data_.last_group_flags[i] = gpu_tree_data_.group_flags[i];
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

