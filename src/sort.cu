#include <cosmictiger/sort.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/particle.hpp>
#include <cosmictiger/timer.hpp>

#define COUNT_BLOCK_SIZE 1024
#define SORT_BLOCK_SIZE 32
#define SORT_OCCUPANCY (16)
#define SORT_GPU_LIMIT 1024*1024
/*
CUDA_KERNEL gpu_sort_kernel1(particle_set parts, int k, int j, size_t begin, size_t end, double xmid, int xdim) {
	const auto& tid = threadIdx.x + blockIdx.x * blockDim.x;
	const size_t n = end - begin;
	//printf( "%i %i\n", k, j);
	for (size_t i = tid; i < n; i += blockDim.x * gridDim.x) {
		const size_t l = i ^ j;
		if (l > i) {
			const auto xi = parts.pos(xdim, begin + i).to_double();
			const auto xl = parts.pos(xdim, begin + l).to_double();
			if ((i & k == 0) && (xi >= xl)) {
				parts.swap(begin + i, begin + l);
			}
			if ((i & k != 0) && (xi < xl)) {
				parts.swap(begin + i, begin + l);
			}
		}
//		__syncthreads();
	}
}

CUDA_KERNEL gpu_sort_kernel2(particle_set parts, size_t begin, size_t end, double xmid, int xdim, size_t* pmid) {
	const auto& tid = threadIdx.x + blockIdx.x * blockDim.x;
	const size_t n = end - begin;
	if (tid == 0) {
		for (size_t i = 0; i < n; i++) {
			printf("%e\n", parts.pos(xdim, begin + i).to_double());
		}
	}
	for (size_t i = tid; i < n; i += blockDim.x * gridDim.x) {
		const auto xi = parts.pos(xdim, begin + i).to_double();
		const auto xl = parts.pos(xdim, begin + i + 1).to_double();
		if (xi < xmid && xl >= xmid) {
			*pmid = xl;
		}
	}
	__syncthreads();
	if (tid == 0) {
		if (parts.pos(xdim, begin) >= xmid) {
			*pmid = begin;
		} else if (parts.pos(xdim, end - 1) < xmid) {
			*pmid = end - 1;
		}
	}
	__syncthreads();
}
*/
size_t cpu_sort_kernel(particle_set parts, size_t begin, size_t end, double xmid, int xdim) {

	size_t lo = begin;
	size_t hi = end;
	fixed32 x(xmid);
	while (lo < hi) {
		if (parts.pos(xdim, lo) >= x) {
			while (lo != hi) {
				hi--;
				if (parts.pos(xdim, hi) < x) {
					parts.swap(lo, hi);
					break;
				}
			}
		}
		lo++;
	}
	return hi;
}

void yield();

size_t sort_particles(particle_set parts, size_t begin, size_t end, double xmid, int xdim, sort_type type) {
	auto pmid = cpu_sort_kernel(parts, begin, end, xmid, xdim);
	/*size_t pmid;
	if (end == begin) {
		pmid = end;
	} else {
		if (type == GPU_SORT) {
			auto stream = get_stream();
			size_t* pmidptr;
			unified_allocator alloc;
			int nblocks = 16 * 46;
			pmidptr = (size_t*) alloc.allocate(sizeof(size_t));
			size_t n = end - begin;
			for (int k = 2; k <= n; k *= 2) {
				for (int j = k / 2; j > 0; j /= 2) {
					gpu_sort_kernel1<<<1,1024,0,stream>>>(parts.get_virtual_particle_set(), k, j, begin,end,xmid,xdim);
					CUDA_CHECK(cudaStreamSynchronize(stream));
				}
			}
			CUDA_CHECK(cudaStreamSynchronize(stream));
			gpu_sort_kernel2<<<1,1024,0,stream>>>(parts.get_virtual_particle_set(), begin,end,xmid,xdim, pmidptr);
			pmid = *pmidptr;
			printf("%li\n", pmid);
			abort();
			alloc.deallocate(pmidptr);
		} else {
			pmid = cpu_sort_kernel(parts, begin, end, xmid, xdim);
		}
	}*/
//	printf("Finished Sort  %li %li %li\n", begin, pmid, end);
	return pmid;
}
