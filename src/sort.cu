#include <cosmictiger/sort.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/particle.hpp>
#include <cosmictiger/timer.hpp>

#define COUNT_BLOCK_SIZE 1024
#define SORT_BLOCK_SIZE 32
#define SORT_OCCUPANCY (16)
#define SORT_GPU_LIMIT (1024*1024)

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

#define SORT_SHMEM_SIZE 512

#define ALL 0xFFFFFFFF

CUDA_KERNEL gpu_sort_kernel(particle_set parts, size_t begin, size_t mid, size_t end, double xmid, int xdim,
		unsigned long long* bottom, unsigned long long*top) {
	const int& tid = threadIdx.x;

	__shared__ array<size_t, SORT_SHMEM_SIZE> lo_indices;

	int lo_count;
	size_t lo = 0;
	size_t hi = mid;
	int tmp;
	while (lo < mid) {
		lo_count = 0;
//		printf( "%i %li %i\n", lo, *bottom, max_add);
		while (lo_count < SORT_SHMEM_SIZE - SORT_BLOCK_SIZE && lo < mid) {
			int max_add = min(SORT_BLOCK_SIZE, SORT_SHMEM_SIZE - lo_count);
			if (tid == 0) {
				lo = atomicAdd(bottom, (unsigned long long) max_add);
			}
			lo = __shfl_sync(ALL, lo, 0);
			if (lo < mid) {
				max_add = min((size_t) max_add, (mid - lo));
				int result;
				if (tid < max_add) {
					result = int(!(parts.pos(xdim, lo + tid).to_double() < xmid));
				} else {
					result = 0;
				}
				int my_index = result;
				int count = result;
				for (int P = SORT_BLOCK_SIZE / 2; P >= 1; P /= 2) {
					count += __shfl_xor_sync(ALL, count, P);
				}
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
				if (result) {
					lo_indices[lo_count + my_index] = lo + tid;
				}
				lo_count += count;
			}
		}
		while (lo_count) {
			int result;
			assert(lo_count<=SORT_SHMEM_SIZE);
			int max_add = min(lo_count, SORT_BLOCK_SIZE);
			if (tid == 0) {
				hi = atomicAdd(top, (unsigned long long) max_add);
			}
			hi = __shfl_sync(ALL, hi, 0);
			max_add = min((size_t) max_add, end - hi);
			if (tid < max_add) {
				//	assert(hi + tid < end);
				result = int(parts.pos(xdim, hi + tid).to_double() < xmid);
			} else {
				result = 0;
			}
			int my_index = result;
			int count = result;
			for (int P = SORT_BLOCK_SIZE / 2; P >= 1; P /= 2) {
				count += __shfl_xor_sync(ALL, count, P);
			}
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
			if (result) {
				const size_t i1 = lo_indices[lo_count - my_index - 1];
				const size_t i2 = hi + tid;
				parts.swap(i1, i2);
			}
			lo_count -= count;
			assert(lo_count >= 0);
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

size_t sort_particles(particle_set parts, size_t begin, size_t end, double xmid, int xdim) {
	size_t pmid;
	static std::atomic<int> kernels_active(0);
	if (end == begin) {
		pmid = end;
	} else {
		if (end - begin >= SORT_GPU_LIMIT) {
			auto stream = get_stream();
			if (end - begin == global().opts.nparts) {
				parts.prepare_sort(stream);
			}
			pmid = begin + gpu_count_particles(parts, begin, end, xmid, xdim, stream);
			CUDA_CHECK(cudaStreamSynchronize(stream));
			unsigned long long* indexes;
			CUDA_MALLOC(indexes, 2);
			const auto nprocs = global().cuda.devices[0].multiProcessorCount;
			indexes[0] = begin;
			indexes[1] = pmid;
			kernels_active++;
//			printf( "%li\n",(int) kernels_active);
			gpu_sort_kernel<<<2*SORT_OCCUPANCY*nprocs,SORT_BLOCK_SIZE>>>(parts, begin, pmid, end, xmid, xdim, indexes + 0, indexes + 1);
			CUDA_CHECK(cudaStreamSynchronize(stream));
			kernels_active--;
			cleanup_stream(stream);
			CUDA_FREE(indexes);
		} else {
			pmid = cpu_sort_kernel(parts, begin, end, xmid, xdim);
		}
	}
//	printf("Finished Sort  %li %li %li\n", begin, pmid, end);
	return pmid;
}
