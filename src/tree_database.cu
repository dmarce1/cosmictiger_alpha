#include <cosmictiger/tree.hpp>
#include <cosmictiger/tree_database.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/memory.hpp>

#include <cmath>
#include <atomic>

static constexpr int chunk_size = 2048;

struct tree_data_t {
	float* radius;
	int8_t* leaf;
	array<fixed32, NDIM>* pos;
	multipole* multi;
	array<tree_ptr,NCHILD>* children;
	int ntrees;
	int nchunks;
};

__managed__ tree_data_t data;

static std::atomic<int> next_chunk;
static int era = 0;

static thread_local int current_alloc;
static thread_local int current_offset;
static thread_local int current_era = -1;

void tree_data_initialize() {
	data.ntrees = global().opts.nparts / global().opts.bucket_size;
	data.ntrees = std::max(1 << ((int) std::ceil(std::log(data.ntrees) / std::log(2)) + 3), 1024 * 1024);
	data.nchunks = data.ntrees / chunk_size;

	CUDA_MALLOC(data.radius, data.ntrees);
	CUDA_MALLOC(data.pos, data.ntrees);
	CUDA_MALLOC(data.leaf, data.ntrees);
	CUDA_MALLOC(data.multi, data.ntrees);
	CUDA_MALLOC(data.children, data.ntrees);

	printf("Allocating %i trees in %i chunks of %i each\n", data.ntrees, data.nchunks, chunk_size);

	tree_data_clear();

}


void tree_data_clear() {
	next_chunk = 0;
	era++;
}

int tree_data_allocate() {
	if( era != current_era) {
		current_era = era;
		current_offset = chunk_size;
	}
	if( current_offset >= chunk_size) {
		current_alloc = chunk_size * next_chunk++;
		current_offset = 0;
	}
	if( current_alloc + chunk_size >= data.ntrees) {
		printf( "Fatal error - tree arena full!\n");
		abort();
	}
	int alloc = current_offset++ + current_alloc;
	if( alloc == 0 ) { // reserve 0 for root
		alloc = tree_data_allocate();
	}
	//printf( "Allocating %i from chunk %i\n", alloc, alloc / chunk_size);
	return alloc;
}

CUDA_EXPORT
float tree_data_get_radius(int i) {
	assert(i>=0);
	assert(i<data.ntrees);
	return data.radius[i];
}

CUDA_EXPORT
void tree_data_set_radius(int i, float r) {
	assert(i>=0);
	assert(i<data.ntrees);
	data.radius[i] = r;
}

CUDA_EXPORT
array<fixed32, NDIM> tree_data_get_pos(int i) {
	assert(i>=0);
	assert(i<data.ntrees);
	return data.pos[i];
}

CUDA_EXPORT
void tree_data_set_pos(int i, const array<fixed32, NDIM>& p) {
	assert(i>=0);
	assert(i<data.ntrees);
	data.pos[i] = p;
}


CUDA_EXPORT
multipole tree_data_get_multi(int i) {
	assert(i>=0);
	assert(i<data.ntrees);
	return data.multi[i];
}

CUDA_EXPORT
void tree_data_set_multi(int i,const multipole& m) {
	assert(i>=0);
	assert(i<data.ntrees);
	data.multi[i] = m;
}



CUDA_EXPORT
bool tree_data_get_isleaf(int i) {
	assert(i>=0);
	assert(i<data.ntrees);
	return data.leaf[i];

}

CUDA_EXPORT
void tree_data_set_isleaf(int i, bool b) {
	assert(i>=0);
	data.leaf[i] = b;

}


CUDA_EXPORT
array<tree_ptr,NCHILD> tree_data_get_children(int i) {
	return data.children[i];
}


CUDA_EXPORT
void tree_data_set_children(int i, const array<tree_ptr,NCHILD>& c) {
	data.children[i] = c;
}

