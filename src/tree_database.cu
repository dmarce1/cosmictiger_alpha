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
	pair<size_t, size_t>* parts;
	array<tree_ptr, NCHILD>* children;
	size_t* active_parts;
	size_t* active_nodes;
	int ntrees;
	int nchunks;
};

__managed__ tree_data_t tree_data_;

static std::atomic<int> next_chunk;
static int era = 0;

static thread_local int current_alloc;
static thread_local int current_offset;
static thread_local int current_era = -1;

void tree_data_initialize() {
	tree_data_.ntrees = global().opts.nparts / global().opts.bucket_size;
	tree_data_.ntrees = std::max(1 << ((int) std::ceil(std::log(tree_data_.ntrees) / std::log(2)) + 3), 1024 * 1024);
	tree_data_.nchunks = tree_data_.ntrees / chunk_size;

	CUDA_MALLOC(tree_data_.radius, tree_data_.ntrees);
	CUDA_MALLOC(tree_data_.pos, tree_data_.ntrees);
	CUDA_MALLOC(tree_data_.leaf, tree_data_.ntrees);
	CUDA_MALLOC(tree_data_.multi, tree_data_.ntrees);
	CUDA_MALLOC(tree_data_.children, tree_data_.ntrees);
	CUDA_MALLOC(tree_data_.parts, tree_data_.ntrees);
	CUDA_MALLOC(tree_data_.active_parts, tree_data_.ntrees);
	CUDA_MALLOC(tree_data_.active_nodes, tree_data_.ntrees);

	for (int i = 0; i < tree_data_.ntrees; i++) {
		new (tree_data_.children + i) array<tree_ptr,NCHILD>();
	}

	printf("Allocating %i trees in %i chunks of %i each\n", tree_data_.ntrees, tree_data_.nchunks, chunk_size);

	tree_data_clear();

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
	if (current_alloc + chunk_size >= tree_data_.ntrees) {
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

CUDA_EXPORT
float tree_data_get_radius(int i) {
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	return tree_data_.radius[i];
}

CUDA_EXPORT
void tree_data_set_radius(int i, float r) {
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	tree_data_.radius[i] = r;
}

CUDA_EXPORT
array<fixed32, NDIM> tree_data_get_pos(int i) {
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	return tree_data_.pos[i];
}

CUDA_EXPORT
void tree_data_set_pos(int i, const array<fixed32, NDIM>& p) {
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	tree_data_.pos[i] = p;
}

CUDA_EXPORT
multipole tree_data_get_multi(int i) {
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	return tree_data_.multi[i];
}

CUDA_EXPORT
void tree_data_set_multi(int i, const multipole& m) {
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	tree_data_.multi[i] = m;
}

CUDA_EXPORT
bool tree_data_get_isleaf(int i) {
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	return tree_data_.leaf[i];

}

CUDA_EXPORT
void tree_data_set_isleaf(int i, bool b) {
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	tree_data_.leaf[i] = b;

}

CUDA_EXPORT
array<tree_ptr, NCHILD> tree_data_get_children(int i) {
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	return tree_data_.children[i];
}

CUDA_EXPORT
void tree_data_set_children(int i, const array<tree_ptr, NCHILD>& c) {
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	tree_data_.children[i] = c;
}

CUDA_EXPORT
pair<size_t, size_t> tree_data_get_parts(int i) {
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	return tree_data_.parts[i];
}

CUDA_EXPORT
void tree_data_set_parts(int i, const pair<size_t, size_t>& p) {
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	tree_data_.parts[i] = p;
}



CUDA_EXPORT
size_t tree_data_get_active_parts(int i) {
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	return tree_data_.active_parts[i];
}

CUDA_EXPORT
void tree_data_set_active_parts(int i, size_t p) {
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	tree_data_.active_parts[i] = p;
}




CUDA_EXPORT
size_t tree_data_get_active_nodes(int i) {
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	return tree_data_.active_nodes[i];
}

CUDA_EXPORT
void tree_data_set_active_nodes(int i, size_t p) {
	assert(i >= 0);
	assert(i < tree_data_.ntrees);
	tree_data_.active_nodes[i] = p;
}


