#include <cosmictiger/hydro_particle.hpp>
#include <cosmictiger/global.hpp>

hydro_particle_set::hydro_particle_set(size_t size, size_t offset) :
		particle_set(size, offset) {
	if (size) {
		CUDA_MALLOC(hptr_, size);
		CUDA_MALLOC(eneptr_, size);
		CUDA_MALLOC(store_, size);
	}

}

CUDA_EXPORT
hydro_particle_set hydro_particle_set::get_virtual_particle_set() const {
	hydro_particle_set parts;
	((particle_set&) parts) = particle_set::get_virtual_particle_set();
	parts.hptr_ = hptr_;
	parts.store_ = store_;
	parts.eneptr_ = eneptr_;
	return parts;
}

size_t hydro_particle_set::sort_range(size_t begin, size_t end, double xm, int xdim) {

	size_t lo = begin;
	size_t hi = end;
	fixed32 xmid(xm);
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
	return hi;
}
