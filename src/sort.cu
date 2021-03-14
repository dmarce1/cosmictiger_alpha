#include <cosmictiger/sort.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/particle.hpp>
#include <cosmictiger/timer.hpp>

#define COUNT_BLOCK_SIZE 1024
#define SORT_BLOCK_SIZE 32
#define SORT_OCCUPANCY (16)
#define COUNT_OCCUPANCY (1)
#define SORT_GPU_LIMIT 64*1024

CUDA_KERNEL count_kernel(particle_set parts, size_t begin, size_t end, double xpos, int xdim, unsigned long long* pmid1,
		unsigned long long* pmid2) {
	const int& tid = threadIdx.x;
	const int& bsz = blockDim.x;
	const int& bid = blockIdx.x;
	const int& gsz = gridDim.x;
	const size_t start = begin + bid * (end - begin) / gsz;
	const size_t stop = begin + (bid + 1) * (end - begin) / gsz;
	__shared__ array<size_t, COUNT_BLOCK_SIZE> local_counts;
	size_t my_count = 0;
	for (size_t i = start + tid; i < stop; i += bsz) {
		if (parts.pos(xdim, i).to_double() < xpos) {
			my_count++;
		}
	}
	local_counts[tid] = my_count;
	__syncthreads();
	for (int P = COUNT_BLOCK_SIZE / 2; P >= 1; P /= 2) {
		if (tid < P) {
			local_counts[tid] += local_counts[tid + P];
		}
		__syncthreads();
	}
	if (tid == 0) {
		atomicAdd(pmid1, local_counts[0]);
		atomicAdd(pmid2, local_counts[0]);
	}
}

#define SORT_SHMEM_SIZE 32

#define ALL 0xFFFFFFFF

CUDA_KERNEL gpu_sort_kernel(particle_set parts, size_t begin, unsigned long long*  mid, size_t end, double xmid, int xdim,
		unsigned long long* bottom, unsigned long long*top) {
	const int& tid = threadIdx.x;
	size_t lo = 0;
	size_t hi = *mid;
	int result = 0;
	int tmp, my_index, index;
	while (lo < *mid || __any_sync(ALL, result)) {
		while (__any_sync(ALL, !result) && lo < *mid) {
			my_index = !result;
			for (int P = SORT_BLOCK_SIZE / 2; P >= 1; P /= 2) {
				my_index += __shfl_down_sync(ALL, my_index, P);
			}
			if (tid == 0) {
				lo = atomicAdd(bottom, my_index);
			}
			lo = __shfl_sync(ALL, lo, 0);
			my_index = !result;
			for (int P = 1; P < SORT_BLOCK_SIZE; P *= 2) {
				tmp = __shfl_up_sync(ALL, my_index, P);
				if (tid >= P) {
					my_index += tmp;
				}
			}
			tmp = __shfl_up_sync(ALL, my_index, 1);
			if (tid > 0) {
				my_index = tmp;
			} else {
				my_index = 0;
			}
			if (!result && lo + my_index < *mid) {
				result = (parts.pos(xdim, lo + my_index).to_double() >= xmid);
				index = lo + my_index;
			}
		}
		while (__any_sync(ALL, result)) {
			my_index = result;
			for (int P = SORT_BLOCK_SIZE / 2; P >= 1; P /= 2) {
				my_index += __shfl_down_sync(ALL, my_index, P);
			}
			if (tid == 0) {
				hi = atomicAdd(top, my_index);
			}
			hi = __shfl_sync(ALL, hi, 0);
			my_index = result;
			for (int P = 1; P < SORT_BLOCK_SIZE; P *= 2) {
				tmp = __shfl_up_sync(ALL, my_index, P);
				if (tid >= P) {
					my_index += tmp;
				}
			}
			tmp = __shfl_up_sync(ALL, my_index, 1);
			if (tid > 0) {
				my_index = tmp;
			} else {
				my_index = 0;
			}
			if (result && parts.pos(xdim, hi + my_index).to_double() < xmid) {
				parts.swap(hi + my_index, index);
				result = 0;
			}
		}
	}
//	if(threadIdx.x == 0 )
//	printf( "%i\n", blockIdx.x);
}

size_t cpu_sort_kernel(particle_set parts, size_t begin, size_t end, double xmid, int xdim) {
	size_t lo = begin;
	size_t hi = end;
	while (lo < hi) {
		if (parts.pos(xdim, lo).to_double() >= xmid) {
			while (lo != hi) {
				hi--;
				if (parts.pos(xdim, hi).to_double() < xmid) {
					parts.swap(lo, hi);
					break;
				}
			}
		}
		lo++;
	}
	return hi;
}

std::function<bool(size_t*)> sort_particles(particle_set parts, size_t begin, size_t end, double xmid, int xdim) {
	auto stream = get_stream();
	const auto nprocs = global().cuda.devices[0].multiProcessorCount;
	unsigned long long* indexes;
	CUDA_MALLOC(indexes, 3);
	indexes[2] = begin;
	indexes[1] = begin;
	indexes[0] = begin;
	int count_blocks = std::max(2 * nprocs * COUNT_OCCUPANCY * (end - begin) / global().opts.nparts, size_t(1));
	int sort_blocks = std::max(2 * nprocs * SORT_OCCUPANCY * (end - begin) / global().opts.nparts, size_t(1));
//	printf( "%i %i\n", count_blocks, sort_blocks);
	count_kernel<<<count_blocks,COUNT_BLOCK_SIZE,0,stream>>>(parts,begin,end,xmid,xdim,indexes+1, indexes + 2);
	gpu_sort_kernel<<<sort_blocks,SORT_BLOCK_SIZE,0,stream>>>(parts, begin, indexes+2, end, xmid, xdim, indexes + 0, indexes + 1);
//	printf("Returning %li %li\n", begin, end);
	return [indexes, stream](size_t* pmidptr) {
		if( cudaStreamQuery(stream)==cudaSuccess) {
			CUDA_CHECK(cudaStreamSynchronize(stream));
			*pmidptr = indexes[2];
			//	printf( "%li\n", *pmidptr);
			CUDA_FREE(indexes);
			cleanup_stream(stream);
			return true;
		} else {
			//		printf( "Sort not  done\n");
			return false;
		}
	};
}

