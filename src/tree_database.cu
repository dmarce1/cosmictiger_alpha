#define TREE_DATABASE_CU
#include <cosmictiger/tree.hpp>
#include <cosmictiger/tree_database.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/memory.hpp>

#include <cmath>
#include <atomic>

static const int min_trees = 1024 * 1024;

static std::atomic<int> next_chunk;

double last_utilization = -1.0;

int hardware_concurrency();

__managed__ int myrank;

template<class T>
void free_if_needed(T** ptr) {
	if (*ptr) {
		CUDA_FREE(*ptr);
		*ptr = nullptr;
	}
}



CUDA_EXPORT int hpx_rank_cuda() {
	return myrank;
}

void tree_data_initialize_kick() {
	myrank = hpx_rank();
	cpu_tree_data_.chunk_size = 1;
	int ntrees;
//	if( last_utilization <= 0.0 ) {
		ntrees = 8 * global().opts.nparts / global().opts.bucket_size / hpx_size();
		cpu_tree_data_.ntrees = ntrees;
		cpu_tree_data_.ntrees = std::max(cpu_tree_data_.ntrees, min_trees);
//	} else {
//		ntrees = cpu_tree_data_.ntrees * last_utilization / 0.85;
//		cpu_tree_data_.ntrees = ntrees;
	//	printf( "%e %i\n", last_utilization, cpu_tree_data_.ntrees);
	//}
	const int target_chunk_size = cpu_tree_data_.ntrees / (16 * OVERSUBSCRIPTION * hardware_concurrency());
	while (cpu_tree_data_.chunk_size < target_chunk_size) {
		cpu_tree_data_.chunk_size *= 2;
	}
	cpu_tree_data_.nchunks = cpu_tree_data_.ntrees / cpu_tree_data_.chunk_size;


//	PRINT("Allocating %i trees in %i chunks of %i each for kick\n", cpu_tree_data_.ntrees, cpu_tree_data_.nchunks,
//			cpu_tree_data_.chunk_size);

	tree_data_free_all_cu();
	CUDA_MALLOC(cpu_tree_data_.proc_range, cpu_tree_data_.ntrees);
	CUDA_MALLOC(cpu_tree_data_.data, cpu_tree_data_.ntrees);
	CUDA_MALLOC(cpu_tree_data_.parts, cpu_tree_data_.ntrees);
	CUDA_MALLOC(cpu_tree_data_.multi, cpu_tree_data_.ntrees);
	CUDA_MALLOC(cpu_tree_data_.active_nodes, cpu_tree_data_.ntrees);
	CUDA_MALLOC(cpu_tree_data_.local_root, cpu_tree_data_.ntrees);
	CUDA_MALLOC(cpu_tree_data_.active_parts, cpu_tree_data_.ntrees);
	tree_data_clear();
/*	for( int i = 0; i < cpu_tree_data_.ntrees; i++) {
		cpu_tree_data_.data[i].children[0].rank = 111111;
		cpu_tree_data_.data[i].children[1].rank = 111111;
	}*/
	cuda_set_device();
	CUDA_CHECK(cudaMemcpyToSymbol(gpu_tree_data_, &cpu_tree_data_, sizeof(cpu_tree_data_)));

}

void tree_data_initialize_groups() {
	myrank = hpx_rank();
	cpu_tree_data_.chunk_size = 1;
	int ntrees;
//	if( last_utilization <= 0.0 ) {
		ntrees = 4 * global().opts.nparts / global().opts.bucket_size / hpx_size();
		cpu_tree_data_.ntrees = ntrees;
		cpu_tree_data_.ntrees = std::max(cpu_tree_data_.ntrees, min_trees);
	//} else {
	//	ntrees = cpu_tree_data_.ntrees * last_utilization / 0.85;
	//	cpu_tree_data_.ntrees = ntrees;
	//}
/*	cpu_tree_data_.ntrees = 1;
	while( cpu_tree_data_.ntrees < ntrees ) {
		cpu_tree_data_.ntrees *= 2;
	}*/
	const int target_chunk_size = cpu_tree_data_.ntrees / (16 * OVERSUBSCRIPTION * hardware_concurrency());
	while (cpu_tree_data_.chunk_size < target_chunk_size) {
		cpu_tree_data_.chunk_size *= 2;
	}
	cpu_tree_data_.nchunks = cpu_tree_data_.ntrees / cpu_tree_data_.chunk_size;


//	PRINT("Allocating %i trees in %i chunks of %i each for group search\n", cpu_tree_data_.ntrees,
//			cpu_tree_data_.nchunks, cpu_tree_data_.chunk_size);

	tree_data_free_all_cu();
	CUDA_MALLOC(cpu_tree_data_.proc_range, cpu_tree_data_.ntrees);
	CUDA_MALLOC(cpu_tree_data_.data, cpu_tree_data_.ntrees);
	CUDA_MALLOC(cpu_tree_data_.parts, cpu_tree_data_.ntrees);
	CUDA_MALLOC(cpu_tree_data_.ranges, cpu_tree_data_.ntrees);
	CUDA_MALLOC(cpu_tree_data_.active_parts, cpu_tree_data_.ntrees);
	CUDA_MALLOC(cpu_tree_data_.local_root, cpu_tree_data_.ntrees);
	CUDA_MALLOC(cpu_tree_data_.active_nodes, cpu_tree_data_.ntrees);

	tree_data_clear();
	cuda_set_device();
	CUDA_CHECK(cudaMemcpyToSymbol(gpu_tree_data_, &cpu_tree_data_, sizeof(cpu_tree_data_)));

}

void tree_data_free_all_cu() {
	free_if_needed(&cpu_tree_data_.data);
	free_if_needed(&cpu_tree_data_.proc_range);
	free_if_needed(&cpu_tree_data_.parts);
	free_if_needed(&cpu_tree_data_.multi);
	free_if_needed(&cpu_tree_data_.ranges);
	free_if_needed(&cpu_tree_data_.active_nodes);
	free_if_needed(&cpu_tree_data_.active_parts);
	free_if_needed(&cpu_tree_data_.local_root);
}

size_t tree_data_bytes_used() {
	size_t use = 0;
	if (cpu_tree_data_.data) {
		use += sizeof(tree_data_t);
	}
	if (cpu_tree_data_.parts) {
		use += sizeof(pair<size_t, size_t> );
	}
	if (cpu_tree_data_.multi) {
		use += sizeof(multipole_pos);
	}
	if (cpu_tree_data_.ranges) {
		use += sizeof(range);
	}
	if (cpu_tree_data_.active_nodes) {
		use += sizeof(size_t);
	}
	if (cpu_tree_data_.active_parts) {
		use += sizeof(size_t);
	}
	use *= cpu_tree_data_.nchunks * cpu_tree_data_.chunk_size;
	return use;
}

void tree_database_set_groups() {
	for (int i = 0; i < cpu_tree_data_.ntrees; i++) {
		cpu_tree_data_.active_parts[i] = cpu_tree_data_.active_nodes[i];
		cpu_tree_data_.active_nodes[i] = 0;
	}
}



void tree_data_clear_cu() {
	last_utilization = (double) next_chunk / cpu_tree_data_.nchunks;
	next_chunk = 0;
	if (cpu_tree_data_.data) {
		for (int i = 0; i < cpu_tree_data_.ntrees; i++) {
			cpu_tree_data_.data[i].children[0].dindex = -1;
		}
	}
}

std::pair<int, int> tree_data_allocate() {
	std::pair<int, int> rc;
	const int chunk = next_chunk++;
	if (chunk >= cpu_tree_data_.nchunks) {
		printf("Fatal error - tree arena full!\n");
		fflush(stdout);
		abort();
	}
	rc.first = chunk * cpu_tree_data_.chunk_size;
	rc.second = rc.first + cpu_tree_data_.chunk_size;
	return rc;
}

double tree_data_use() {
	return (double) next_chunk / (double) cpu_tree_data_.nchunks;
}

