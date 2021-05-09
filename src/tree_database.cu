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


void tree_data_free_all_cu() {
	free_if_needed(&cpu_tree_data().data);
	free_if_needed(&cpu_tree_data().parts);
	free_if_needed(&cpu_tree_data().multi);
	free_if_needed(&cpu_tree_data().sph_ranges);
	free_if_needed(&cpu_tree_data().ranges);
	free_if_needed(&cpu_tree_data().active_nodes);
	free_if_needed(&cpu_tree_data().active_parts);
	free_if_needed(&cpu_tree_data().all_local);
}

void tree_data_initialize_kick() {

	cpu_tree_data().chunk_size = 1;
	cpu_tree_data().ntrees = 5 * global().opts.nparts / global().opts.bucket_size;
	if (global().opts.sph) {
		cpu_tree_data().ntrees *= 2;
	}
	cpu_tree_data().ntrees = std::max(cpu_tree_data().ntrees, min_trees);
	const int target_chunk_size = cpu_tree_data().ntrees / (16 * OVERSUBSCRIPTION * hardware_concurrency());
	while (cpu_tree_data().chunk_size < target_chunk_size) {
		cpu_tree_data().chunk_size *= 2;
	}
	cpu_tree_data().nchunks = cpu_tree_data().ntrees / cpu_tree_data().chunk_size;

//	CUDA_CHECK(cudaMemAdvise(&cpu_tree_data(), sizeof(cpu_tree_data()), cudaMemAdviseSetReadMostly, 0));

	printf("Allocating %i trees in %i chunks of %i each for kick\n", cpu_tree_data().ntrees, cpu_tree_data().nchunks,
			cpu_tree_data().chunk_size);

	tree_data_free_all_cu();
	CUDA_MALLOC(cpu_tree_data().data, cpu_tree_data().ntrees);
	CUDA_MALLOC(cpu_tree_data().parts, cpu_tree_data().ntrees);
	CUDA_MALLOC(cpu_tree_data().multi, cpu_tree_data().ntrees);
	CUDA_MALLOC(cpu_tree_data().active_nodes, cpu_tree_data().ntrees);
	CUDA_MALLOC(cpu_tree_data().active_parts, cpu_tree_data().ntrees);
	CUDA_MALLOC(cpu_tree_data().all_local, cpu_tree_data().ntrees);
	if (global().opts.sph) {
		CUDA_MALLOC(cpu_tree_data().ranges, cpu_tree_data().ntrees);
		CUDA_MALLOC(cpu_tree_data().sph_ranges, cpu_tree_data().ntrees);
	}
	tree_data_clear();

	CUDA_CHECK(cudaMemcpyToSymbol(gpu_tree_data_, &cpu_tree_data(), sizeof(cpu_tree_data())));
	printf( "Done allocating for kick 3\n");

}

void tree_data_initialize_groups() {
	cpu_tree_data().chunk_size = 1;
	cpu_tree_data().ntrees = 4 * global().opts.nparts / GROUP_BUCKET_SIZE;
	cpu_tree_data().ntrees = std::max(cpu_tree_data().ntrees, min_trees);
	const int target_chunk_size = cpu_tree_data().ntrees / (16 * OVERSUBSCRIPTION * hardware_concurrency());
	while (cpu_tree_data().chunk_size < target_chunk_size) {
		cpu_tree_data().chunk_size *= 2;
	}
	cpu_tree_data().nchunks = cpu_tree_data().ntrees / cpu_tree_data().chunk_size;

//	CUDA_CHECK(cudaMemAdvise(&cpu_tree_data(), sizeof(cpu_tree_data()), cudaMemAdviseSetReadMostly, 0));

	printf("Allocating %i trees in %i chunks of %i each for group search\n", cpu_tree_data().ntrees,
			cpu_tree_data().nchunks, cpu_tree_data().chunk_size);

	tree_data_free_all_cu();
	CUDA_MALLOC(cpu_tree_data().data, cpu_tree_data().ntrees);
	CUDA_MALLOC(cpu_tree_data().parts, cpu_tree_data().ntrees);
	CUDA_MALLOC(cpu_tree_data().ranges, cpu_tree_data().ntrees);
	CUDA_MALLOC(cpu_tree_data().active_parts, cpu_tree_data().ntrees);
	CUDA_MALLOC(cpu_tree_data().active_nodes, cpu_tree_data().ntrees);
	CUDA_MALLOC(cpu_tree_data().all_local, cpu_tree_data().ntrees);

	tree_data_clear();

	CUDA_CHECK(cudaMemcpyToSymbol(gpu_tree_data_, &cpu_tree_data(), sizeof(cpu_tree_data())));

}

size_t tree_data_bytes_used() {
	size_t use = 0;
	if (cpu_tree_data().data) {
		use += sizeof(tree_data_t);
	}
	if (cpu_tree_data().parts) {
		use += sizeof(pair<size_t, size_t> );
	}
	if (cpu_tree_data().multi) {
		use += sizeof(multipole_pos);
	}
	if (cpu_tree_data().ranges) {
		use += sizeof(range);
	}
	if (cpu_tree_data().active_nodes) {
		use += sizeof(size_t);
	}
	if (cpu_tree_data().active_parts) {
		use += sizeof(size_t);
	}
	if (cpu_tree_data().all_local) {
		use += sizeof(bool);
	}
	use *= cpu_tree_data().nchunks * cpu_tree_data().chunk_size;
	return use;
}



void tree_data_set_groups_cu() {
	for (int i = 0; i < cpu_tree_data().ntrees; i++) {
		cpu_tree_data().active_parts[i] = cpu_tree_data().active_nodes[i];
		cpu_tree_data().active_nodes[i] = 0;
	}
}

void tree_database_set_readonly() {
#ifdef USE_READMOSTLY
	CUDA_CHECK(cudaMemAdvise(cpu_tree_data().data, cpu_tree_data().ntrees, cudaMemAdviseSetReadMostly, 0));
	CUDA_CHECK(cudaMemAdvise(cpu_tree_data().active_nodes, cpu_tree_data().ntrees, cudaMemAdviseSetReadMostly, 0));
#endif
}

void tree_database_unset_readonly() {
#ifdef USE_READMOSTLY
	CUDA_CHECK(cudaMemAdvise(cpu_tree_data().data, cpu_tree_data().ntrees, cudaMemAdviseUnsetReadMostly, 0));
	CUDA_CHECK(cudaMemAdvise(cpu_tree_data().active_nodes, cpu_tree_data().ntrees, cudaMemAdviseUnsetReadMostly, 0));
#endif
}


void tree_data_clear() {
	tree_cache_clear();
	next_chunk = 0;
}

std::pair<int, int> tree_data_allocate() {
	std::pair<int, int> rc;
	const int chunk = next_chunk++;
	if (chunk >= cpu_tree_data().nchunks) {
		printf("Fatal error - tree arena full!\n");
		abort();
	}
	rc.first = chunk * cpu_tree_data().chunk_size;
	rc.second = rc.first + cpu_tree_data().chunk_size;
	return rc;
}

double tree_data_use() {
	return (double) next_chunk / (double) cpu_tree_data().nchunks;
}

