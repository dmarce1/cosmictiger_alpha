#include <cosmictiger/particle.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/global.hpp>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>

CUDA_EXPORT bool operator<(const particle& a, const particle& b) {
	for (int dim = 0; dim < NDIM; dim++) {
		if (a.x[dim].to_double() < b.x[dim].to_double()) {
			return true;
		} else if (a.x[dim].to_double() > b.x[dim].to_double()) {
			return false;
		}
	}
	return false;
}

void particle_set::sort_parts(particle* begin, particle* end) {
	cuda_set_device();
	thrust::sort(thrust::device, begin, end);
}

void particle_set::sort_indices(part_int* begin, part_int* end) {
	cuda_set_device();
	thrust::sort(thrust::device, begin, end);
}

void particle_set::generate_random(int seed) {
	cuda_set_device();
	if (size_) {
		cudaFuncAttributes attribs;
		CUDA_CHECK(cudaFuncGetAttributes(&attribs, generate_random_vectors));
		int num_threads = attribs.maxThreadsPerBlock;
		int num_blocks;
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, generate_random_vectors, num_threads, 0));
		num_blocks *= global().cuda.devices[0].multiProcessorCount;
		generate_random_vectors<<<num_blocks,num_threads>>>(xptr_[0],xptr_[1],xptr_[2],size_,seed);
		CUDA_CHECK(cudaDeviceSynchronize());

		for (int i = 0; i < size_; i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				vel(0, i) = 0.f;
				vel(1, i) = 0.f;
				vel(2, i) = 0.f;
			}
			set_rung(0, i);
		}
	}
}

#define COUNT_PARTS_KERNEL_SIZE 1024

__global__ void count_parts_kernel(fixed32* x, fixed32 xmid, part_int part_count, part_int* count) {
	const part_int tid = threadIdx.x;
	const part_int bid = blockIdx.x;
	const part_int gsz = gridDim.x;
	const part_int start = size_t(bid) * size_t(part_count) / size_t(gsz);
	const part_int stop = size_t(bid + 1) * size_t(part_count) / size_t(gsz);
	__shared__ part_int mycounts[COUNT_PARTS_KERNEL_SIZE];
	mycounts[tid] = 0;
	for (part_int i = start + tid; i < stop; i += COUNT_PARTS_KERNEL_SIZE) {
		mycounts[tid] += part_int(x[i] < xmid);
	}
	for (int P = COUNT_PARTS_KERNEL_SIZE / 2; P >= 1; P /= 2) {
		__syncthreads();
		if (tid < P) {
			mycounts[tid] += mycounts[tid + P];
		}
	}
	if (tid == 0) {
		atomicAdd(count, mycounts[0]);
	}
}

__global__ void find_range_kernel(fixed32* x, part_int part_count, fixed32* xmin, fixed32* xmax) {
	const part_int tid = threadIdx.x;
	const part_int bid = blockIdx.x;
	const part_int gsz = gridDim.x;
	const part_int start = size_t(bid) * size_t(part_count) / size_t(gsz);
	const part_int stop = size_t(bid + 1) * size_t(part_count) / size_t(gsz);
	__shared__ fixed32 mymax[COUNT_PARTS_KERNEL_SIZE];
	__shared__ fixed32 mymin[COUNT_PARTS_KERNEL_SIZE];
	mymax[tid] = 0.0;
	mymin[tid] = fixed32::max();
	for (part_int i = start + tid; i < stop; i += COUNT_PARTS_KERNEL_SIZE) {
		mymax[tid] = max(mymax[tid], x[i]);
		mymin[tid] = min(mymin[tid], x[i]);
	}
	for (int P = COUNT_PARTS_KERNEL_SIZE / 2; P >= 1; P /= 2) {
		__syncthreads();
		if (tid < P) {
			mymax[tid] = max(mymax[tid], mymax[tid + P]);
			mymin[tid] = min(mymin[tid], mymax[tid + P]);
		}
	}
	if (tid == 0) {
		atomicMax((unsigned *) xmax, mymax[0].raw());
		atomicMin((unsigned *) xmin, mymin[0].raw());
	}
}

part_int particle_set::count_parts_below(part_int b, part_int e, int xdim, fixed32 xmid) const {
	part_int* count;
	unified_allocator alloc;
	int num_blocks;
	count = (part_int*) alloc.allocate(sizeof(part_int));
	CUDA_CHECK(
			cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, count_parts_kernel, COUNT_PARTS_KERNEL_SIZE, COUNT_PARTS_KERNEL_SIZE*sizeof(part_int)));
	num_blocks *= global().cuda.devices[0].multiProcessorCount;
	num_blocks = int(double(num_blocks) * double(e - b) / double(size_));
	*count = 0;
	if (num_blocks == 0) {
		const auto& x = xptr_[xdim];
		for (part_int i = b; i < e; i++) {
			if (x[i] < xmid) {
				(*count)++;
			}
		}
	} else {
		cuda_set_device();
		count_parts_kernel<<<num_blocks,COUNT_PARTS_KERNEL_SIZE>>>(xptr_[xdim] + b, xmid, e - b, count);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	const int rc = *count;
	alloc.deallocate(count);
	return rc;
}

std::pair<fixed32, fixed32> particle_set::find_range(part_int b, part_int e, int xdim) const {
	fixed32* xmax;
	fixed32* xmin;
	unified_allocator alloc;
	int num_blocks;
	xmax = (fixed32*) alloc.allocate(sizeof(fixed32));
	xmin = (fixed32*) alloc.allocate(sizeof(fixed32));
	CUDA_CHECK(
			cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, find_range_kernel, COUNT_PARTS_KERNEL_SIZE, 2*COUNT_PARTS_KERNEL_SIZE*sizeof(fixed32)));
	num_blocks *= global().cuda.devices[0].multiProcessorCount;
	num_blocks = int(double(num_blocks) * double(e - b) / double(size_));
	*xmin = fixed32::max();
	*xmax = 0.0;
	if (num_blocks == 0) {
		const auto& x = xptr_[xdim];
		for (part_int i = b; i < e; i++) {
			*xmax = max(*xmax, x[i]);
			*xmin = min(*xmin, x[i]);
		}
	} else {
		cuda_set_device();
		find_range_kernel<<<num_blocks,COUNT_PARTS_KERNEL_SIZE>>>(xptr_[xdim] + b, e - b, xmin, xmax);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	const auto rc = std::make_pair(*xmin, *xmax);
	alloc.deallocate(xmax);
	alloc.deallocate(xmin);
	return rc;
}

fixed32 particle_set::find_middle(part_int b, part_int e, int xdim) const {
	fixed32 xmax, xmin, xmid;
	part_int target = (e - b) / 2;
	auto rng = find_range(b, e, xdim);
	xmax = rng.second;
	xmin = rng.first;
	do {
		xmid = (xmax.to_double() + xmin.to_double()) / 2.0;
		part_int cnt = count_parts_below(b, e, xdim, xmid);
		if (cnt < target) {
			xmin = xmid;
		} else if (cnt > target) {
			xmax = xmid;
		} else {
			xmin = xmax = xmid;
		}
	} while (xmax - xmin > fixed32::min());
	return xmid;
}
