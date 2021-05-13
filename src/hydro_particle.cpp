#include <cosmictiger/hydro_particle.hpp>
#include <cosmictiger/global.hpp>

hydro_particle_set::hydro_particle_set(size_t size, size_t offset) :
		particle_set(size, offset) {
	if (size) {
		CUDA_MALLOC(hptr_, size);
		CUDA_MALLOC(eneptr_, size);
		CUDA_MALLOC(store_, size);
		CUDA_MALLOC(semi_, size);
	}

}

CUDA_EXPORT
hydro_particle_set hydro_particle_set::get_virtual_particle_set() const {
	hydro_particle_set parts;
	((particle_set&) parts) = particle_set::get_virtual_particle_set();
	parts.hptr_ = hptr_;
	parts.store_ = store_;
	parts.eneptr_ = eneptr_;
	parts.semi_ = semi_;
	return parts;
}

size_t hydro_particle_set::sort_range(size_t begin, size_t end, range box, int xdim, size_t bucket_size) {

	size_t lo = begin;
	size_t hi = end;
	fixed32 xmid = (box.begin[xdim] + box.end[xdim])/2.0;
	auto& xptr_dim = xptr_[xdim];
	auto& x = xptr_[0];
	auto& y = xptr_[1];
	auto& z = xptr_[2];
	const bool groups = global().opts.groups;
	while (lo < hi) {
		if (xptr_dim[lo] >= xmid) {
			while (lo != hi) {
				hi--;
				if (xptr_dim[hi] < xmid) {
					std::swap(x[hi], x[lo]);
					std::swap(y[hi], y[lo]);
					std::swap(z[hi], z[lo]);
					std::swap(uptr_[hi][0], uptr_[lo][0]);
					std::swap(uptr_[hi][1], uptr_[lo][1]);
					std::swap(uptr_[hi][2], uptr_[lo][2]);
					std::swap(rptr_[hi], rptr_[lo]);
					std::swap(hptr_[hi], hptr_[lo]);
					std::swap(eneptr_[hi], eneptr_[lo]);
					if (groups) {
						std::swap(idptr_[hi], idptr_[lo]);
					}
					break;
				}
			}
		}
		lo++;
	}
	if( bucket_size && end - begin > bucket_size ) {
		range left, right;
		int next_dim = (xdim + 1) % NDIM;
		left = right = box;
		left.end[xdim] = xmid.to_float();
		right.begin[xdim] = xmid.to_float();
		const auto left_func = [this,begin,hi,left,next_dim,bucket_size](){
			sort_range(begin,hi,left,next_dim,bucket_size);
		};
		const auto right_func = [this,end,hi,right,next_dim,bucket_size](){
			sort_range(hi,end,right,next_dim,bucket_size);
		};
		if( end - begin > 65536 ) {
			hpx::future<void> futl, futr;
			futl  = hpx::async(left_func);
			futr  = hpx::async(right_func);
			futl.get();
			futr.get();
		} else {
			left_func();
			right_func();
		}
	}
	return hi;
}
