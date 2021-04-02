#include <cosmictiger/trees.hpp>
#include <cosmictiger/tree.hpp>

#include <cmath>

trees trees_allocator::tree_database;
std::atomic<int> next_chunk_;

void trees::init() {
	const int nparts = global().opts.nparts;
	const int bucket_size = global().opts.bucket_size;
	ntrees_ = std::max(8 * std::max(nparts / bucket_size, (int) 1), 1024 * 1024);
	nchunks_ = 8 * (hpx::thread::hardware_concurrency() * OVERSUBSCRIPTION);
	ntrees_ = ((ntrees_ - 1) / nchunks_ + 1) * nchunks_;
	chunk_size_ = ntrees_ / nchunks_;
	printf("Allocating trees - %i trees in %i chunks of %i size\n", ntrees_, nchunks_, chunk_size_);
	CUDA_MALLOC(multi_data_, ntrees_);
	CUDA_MALLOC(child_data_, ntrees_);
	CUDA_MALLOC(parts_data_, ntrees_);
	CUDA_MALLOC(crit_data_, ntrees_);
	get_cuda_trees_database() = *this;
	initialized = true;
}

int trees::get_chunk() {
	assert(initialized);
	int this_chunk = next_chunk_++;
	if (this_chunk >= nchunks_) {
		printf("FATAL ERROR: Trees out of available branches!\n");
		abort();
	}
	return chunk_size_ * this_chunk;
}

void trees::clear() {
	assert(initialized);
	next_chunk_ = 0;
}

int trees::chunk_size() const {
	assert(initialized);
	return chunk_size_;
}

