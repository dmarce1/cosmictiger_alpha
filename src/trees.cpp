#include <cosmictiger/trees.hpp>
#include <cosmictiger/tree.hpp>

#include <cmath>

trees trees_allocator::tree_database;

void trees::init() {
	const int nparts = global().opts.nparts;
	const int bucket_size = global().opts.bucket_size;
	next_chunk_ = 0;
	ntrees_ = std::max(4 * std::max(nparts / bucket_size, (int) 1), 1024 * 1024);
	nchunks_ = 8 * (hpx::thread::hardware_concurrency() * OVERSUBSCRIPTION);
	ntrees_ = ((ntrees_ - 1) / nchunks_ + 1) * nchunks_;
	chunk_size_ = ntrees_ / nchunks_;
	printf("Allocating trees - %li trees in %li chunks of %li size\n", ntrees_, nchunks_, chunk_size_);
	const int total_element_sz = sizeof(multipole_pos) + sizeof(children_type) + sizeof(parts_type);
	const int total_sz = total_element_sz * ntrees_;
	char* cdata;
	CUDA_MALLOC(cdata, total_sz);
	data_ = (void*) cdata;
	multi_data_ = (multipole_pos*) data_;
	child_data_ = (children_type*) (((char*) data_) + sizeof(multipole_pos) * ntrees_);
	parts_data_ = (parts_type*) (((char*) child_data_) + sizeof(children_type) * ntrees_);
	crit_data_ = (multi_crit*) (((char*) parts_data_) + sizeof(parts_type) * ntrees_);
}

int trees::get_chunk() {
	if (next_chunk_ == nchunks_) {
		printf("FATAL ERROR: Trees out of available branches!\n");
		abort();
	}
	return chunk_size_ * next_chunk_++;
}

void trees::clear() {
	next_chunk_ = 0;
}

int trees::chunk_size() const {
	return chunk_size_;
}

