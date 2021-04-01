#include <cosmictiger/trees.hpp>
#include <cosmictiger/tree.hpp>

#include <cmath>

trees::trees() :
		next_chunk_(0) {
	const size_t nparts = global().opts.nparts;
	const size_t bucket_size = global().opts.bucket_size;

	ntrees_ = 2 * std::min(nparts / bucket_size, (size_t) 1);
	nchunks_ = 1024;
	ntrees_ = ((ntrees_ - 1) / nchunks_ + 1) * nchunks_;
	chunk_size_ = nchunks_ / ntrees_;
	printf("Allocating trees - %li trees in %li chunks of %li size\n", ntrees_, nchunks_, chunk_size_);
	const size_t total_element_sz = sizeof(multipole_pos) + sizeof(children_type) + sizeof(parts_type);
	const size_t total_sz = total_element_sz * ntrees_;
	char* cdata;
	CUDA_MALLOC(cdata, total_sz);
	data_ = (void*) cdata;
	multi_data_ = (multipole_pos*) data_;
	child_data_ = (children_type*) (((char*) data_) + sizeof(multipole_pos) * ntrees_);
	parts_data_ = (parts_type*) (((char*) child_data_) + sizeof(children_type) * ntrees_);
	crit_data_ = (multi_crit*) (((char*) parts_data_) + sizeof(parts_type) * ntrees_);
}

size_t trees::get_chunk() {
	if (next_chunk_ == nchunks_) {
		printf("FATAL ERROR: Trees out of available branches!\n");
		abort();
	}
	return next_chunk_++;
}

void trees::clear() {
	next_chunk_ = 0;
}

size_t trees::chunk_size() const {
	return chunk_size_;
}

trees::~trees() {
	CUDA_FREE(data_);
}
