#include <cosmictiger/domain.hpp>
#include <cosmictiger/hpx.hpp>

void domain_bounds::create_uniform_bounds(range box, int nbegin, int nend) {
	int nmid = (nbegin + nend) / 2;
	double w1 = (double) (nmid - nbegin) / (double) (nend - nbegin);
	double w0 = 1.0 - w1;
	double max_span = 0.0;
	for (int dim = 0; dim < NDIM; dim++) {
		const auto span = box.end[dim] - box.begin[dim];
		if (max_span < span) {
			max_span = span;
			xdim = dim;
		}
	}
	bound = w0 * box.begin[xdim] + w1 * box.end[xdim];
	if (nend - nbegin == 1) {
		left = right = nullptr;
	} else {
		range box_left, box_right;
		box_left = box_right = box;
		box_left.end[xdim] = box_right.begin[xdim] = bound.to_double();
		CUDA_MALLOC(left,1);
		CUDA_MALLOC(right,1);
		left->create_uniform_bounds(box_left, nbegin, nmid);
		right->create_uniform_bounds(box_right, nmid, nend);
	}
}

void domain_bounds::create_uniform_bounds() {
	range box;
	for (int dim = 0; dim < NDIM; dim++) {
		box.begin[dim] = 0.0;
		box.end[dim] = 1.0;
	}
	create_uniform_bounds(box, 0, hpx_size());
}

void domain_bounds::destroy() {
	if (left) {
		unified_allocator alloc;
		left->destroy();
		right->destroy();
		CUDA_FREE(left);
		CUDA_FREE(right);
	}
}

CUDA_EXPORT int domain_bounds::find_proc(const array<fixed32, NDIM>& x, int nbegin, int nend) const {
	if (nbegin == -1) {
		nbegin = 0;
		nend = hpx_size();
	}
	int nmid = (nbegin + nend) / 2;
	if (nend - nbegin <= 1) {
		return nbegin;
	} else {
		if (x[xdim] < bound) {
			return left->find_proc(x, nbegin, nmid);
		} else {
			return right->find_proc(x, nmid, nend);
		}
	}
}

CUDA_EXPORT range domain_bounds::find_proc_range(int proc) const {
	range box;
	for (int dim = 0; dim < NDIM; dim++) {
		box.begin[dim] = 0.0;
		box.end[dim] = 1.0;
	}
	return find_proc_range(proc, 0, hpx_size(), box);
}

CUDA_EXPORT range domain_bounds::find_proc_range(int proc, int nbegin, int nend, range box) const {
	if (nend - nbegin <= 1) {
		return box;
	} else {
		const int nmid = (nbegin + nend) / 2;
		range next_box = box;
		if (proc < nmid) {
			next_box.end[xdim] = bound.to_double();
			return left->find_proc_range(proc, nbegin, nmid, next_box);
		} else {
			next_box.begin[xdim] = bound.to_double();
			return right->find_proc_range(proc, nmid, nend, next_box);
		}
	}

}

CUDA_EXPORT range domain_bounds::find_range(pair<int, int> rng) const {
	range box;
	for (int dim = 0; dim < NDIM; dim++) {
		box.begin[dim] = 0.0;
		box.end[dim] = 1.0;
	}
	return find_range(rng, 0, hpx_size(), box);
}

CUDA_EXPORT range domain_bounds::find_range(pair<int, int> rng, int nbegin, int nend, range box) const {
	if (rng.first == nbegin && rng.second == nend) {
		return box;
	} else if (nend - nbegin > 1) {
		const int nmid = (nbegin + nend) / 2;
		range next_box = box;
		if (rng.second <= nmid) {
			next_box.end[xdim] = bound.to_double();
			return left->find_range(rng, nbegin, nmid, next_box);
		} else {
			next_box.begin[xdim] = bound.to_double();
			return right->find_range(rng, nmid, nend, next_box);
		}
	} else {
		ERROR()
		;
		return range();
	}

}

