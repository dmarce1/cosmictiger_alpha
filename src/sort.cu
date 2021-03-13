#include <cosmictiger/sort.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/particle.hpp>
#include <cosmictiger/timer.hpp>

#define COUNT_BLOCK_SIZE 1024
#define SORT_BLOCK_SIZE 32
#define SORT_OCCUPANCY (16)
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

#define SORT_SHMEM_SIZE 32

#define ALL 0xFFFFFFFF

CUDA_KERNEL gpu_sort_kernel(particle_set parts, size_t begin, size_t mid, size_t end, double xmid, int xdim,
		unsigned long long* bottom, unsigned long long*top) {
	const int& tid = threadIdx.x;
	size_t lo = 0;
	size_t hi = mid;
	int result = 0;
	int tmp, my_index, index;
	while (lo < mid || __any_sync(ALL, result)) {
		if (__any_sync(ALL, !result) && lo < mid) {
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
			if (!result && lo + my_index < mid) {
				result = (parts.pos(xdim, lo + my_index).to_double() >= xmid);
				index = lo + my_index;
			}
		}
		if (__any_sync(ALL, result)) {
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
			const auto nprocs = global().cuda.devices[0].multiProcessorCount;
			indexes[0] = begin;
			indexes[1] = pmid;
			gpu_sort_kernel<<<SORT_OCCUPANCY*nprocs,SORT_BLOCK_SIZE>>>(parts, begin, pmid, end, xmid, xdim, indexes + 0, indexes + 1);
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
