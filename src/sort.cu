#include <cosmictiger/sort.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/particle.hpp>
#include <cosmictiger/timer.hpp>

#define COUNT_BLOCK_SIZE 1024
#define SORT_BLOCK_SIZE 32
#define SORT_OCCUPANCY (1)
#define SORT_GPU_LIMIT 1024*1024

CUDA_KERNEL count_kernel(particle_set parts, size_t begin, size_t end, double xpos, int xdim, size_t* counts) {
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
	counts[bid] = local_counts[0];
}

#define SORT_SHMEM_SIZE 2048

#define ALL 0xFFFFFFFF

CUDA_KERNEL gpu_sort_kernel(particle_set parts, size_t begin, size_t mid, size_t end, double xmid, int xdim,
		unsigned long long* bottom, unsigned long long*top) {
	__shared__ size_t swapa[SORT_SHMEM_SIZE];
	__shared__ size_t swapb[SORT_SHMEM_SIZE];

	const int& tid = threadIdx.x;
	const int& bid = blockIdx.x;
	const int& gsz = gridDim.x;
	const size_t start = begin + bid * (mid - begin) / gsz;
	const size_t stop = begin + (bid + 1) * (mid - begin) / gsz;
	size_t lo = start;
	size_t hi = mid;
	int result = 0;
	int count;
	int swap_count = 0;
	int tmp, my_index, lo_index;
	while (lo < stop || __any_sync(ALL, result)) {
		while ((count = __popc(__ballot_sync(ALL, !result))) > SORT_BLOCK_SIZE / 2 && lo < stop) {
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
			if (!result && lo + my_index < stop) {
				result = (parts.pos(xdim, lo + my_index).to_double() >= xmid);
				lo_index = lo + my_index;
			}
			lo += count;
		}
		while ((count = __popc(__ballot_sync(ALL, result))) > SORT_BLOCK_SIZE / 2 || lo >= stop) {
			my_index = result;
			if (tid == 0) {
				hi = atomicAdd(top, count);
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
			int found_swap = 0;
			int hi_index = hi + my_index;
			if (result && parts.pos(xdim, hi + my_index).to_double() < xmid) {
				result = 0;
				found_swap = 1;
			}
			my_index = found_swap;
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
			if (found_swap) {
				swapa[swap_count + my_index] = hi_index;
				swapb[swap_count + my_index] = lo_index;
			}
			swap_count += __popc(__ballot_sync(ALL, found_swap));
		}
		int max_remaining = __popc(__ballot_sync(ALL, !result));
		if (swap_count >= SORT_SHMEM_SIZE - max_remaining || lo >= stop) {
			for (int i = tid; i < swap_count; i += SORT_BLOCK_SIZE) {
				const auto a = swapa[i];
				const auto b = swapb[i];
				for (int dim = 0; dim < NDIM; dim++) {
					parts.swap_pos(dim, a, b);
					parts.swap_vel(dim, a, b);
				}
				parts.swap_rung(a, b);
			}
			swap_count = 0;
		}
	}
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

size_t gpu_count_particles(particle_set parts, size_t begin, size_t end, double xpos, int xdim, cudaStream_t stream) {
	const auto nparts = global().opts.nparts;
	const auto nprocs = global().cuda.devices[0].multiProcessorCount;
	const auto mycount = end - begin;
	int nchunks = std::max(1, (int) (mycount * nprocs / nparts));
	size_t* counts;
	CUDA_MALLOC(counts, nchunks);
	count_kernel<<<nchunks,COUNT_BLOCK_SIZE,0,stream>>>(parts,begin,end,xpos,xdim,counts);
	CUDA_CHECK(cudaStreamSynchronize(stream));
	size_t count = 0;
	for (int i = 0; i < nchunks; i++) {
		count += counts[i];
	}
	CUDA_FREE(counts);
	return count;
}

size_t cpu_count_particles(particle_set parts, size_t begin, size_t end, double xpos, int xdim) {
	size_t count = 0;
	for (size_t i = begin; i < end; i++) {
		const auto x = parts.pos(xdim, i).to_double();
		if (x < xpos) {
			count++;
		}
	}
	return count;
}

void yield();

size_t sort_particles(particle_set parts, size_t begin, size_t end, double xmid, int xdim, sort_type type) {
	size_t pmid;
	if (end == begin) {
		pmid = end;
	} else {
		if (type == GPU_SORT) {
			auto stream = get_stream();
			pmid = begin + gpu_count_particles(parts, begin, end, xmid, xdim, stream);
			CUDA_CHECK(cudaStreamSynchronize(stream));
			unsigned long long* indexes;
			CUDA_MALLOC(indexes, 2);
			const auto nprocs = 2 * SORT_OCCUPANCY * global().cuda.devices[0].multiProcessorCount;
			indexes[0] = begin;
			indexes[1] = pmid;
			gpu_sort_kernel<<<nprocs,SORT_BLOCK_SIZE>>>(parts, begin, pmid, end, xmid, xdim, indexes + 0, indexes + 1);
			yield();
			CUDA_CHECK(cudaStreamSynchronize(stream));
			cleanup_stream(stream);
			CUDA_FREE(indexes);
		} else {
			pmid = cpu_sort_kernel(parts, begin, end, xmid, xdim);
		}
	}
//	printf("Finished Sort  %li %li %li\n", begin, pmid, end);
	return pmid;
}
