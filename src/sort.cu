#include <cosmictiger/sort.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/particle.hpp>

#define COUNT_BLOCK_SIZE 1024
#define SORT_BLOCK_SIZE 32
#define SORT_OCCUPANCY (16*128)

CUDA_KERNEL count_kernel(particle_set parts, size_t begin, size_t end, fixed32 xpos, int xdim, size_t* counts) {
	const int& tid = threadIdx.x;
	const int& bsz = blockDim.x;
	const int& bid = blockIdx.x;
	const int& gsz = gridDim.x;
	const size_t start = begin + bid * (end - begin) / gsz;
	const size_t stop = begin + (bid + 1) * (end - begin) / gsz;
	__shared__ array<size_t, COUNT_BLOCK_SIZE> local_counts;
	size_t my_count = 0;
	for (size_t i = start + tid; i < stop; i += bsz) {
		if (parts.pos(xdim, i) < xpos) {
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

CUDA_KERNEL sort_kernel(particle_set parts, size_t begin, size_t end, fixed32 xmid, int xdim,
		unsigned long long* bottom) {
	const int& tid = threadIdx.x;
	const int& bsz = blockDim.x;
	const int& bid = blockIdx.x;
	const int& gsz = gridDim.x;
	const size_t mid = (begin + end) / 2;
	const size_t start = mid + bid * (end - mid) / gsz;
	const size_t stop = mid + (bid + 1) * (end - mid) / gsz;
	for (size_t i = tid + start; i < stop; i += SORT_BLOCK_SIZE) {
		bool found_swap = (parts.pos(xdim, i) < xmid);
		while (__any_sync(0xFFFFFFFF, found_swap)) {
			int my_index = found_swap;
			int count = found_swap;
			for (int P = SORT_BLOCK_SIZE / 2; P >= 1; P /= 2) {
				count += __shfl_down_sync(0xFFFFFFFF, count, P);
			}
			for (int P = 1; P < SORT_BLOCK_SIZE; P *= 2) {
				int tmp = __shfl_up_sync(0xFFFFFFFF, my_index, P);
				if (tid >= P) {
					my_index += tmp;
				}
			}
			my_index = __shfl_up_sync(0xFFFFFFFF, my_index, 1);
			size_t base_index;
			if (tid == 0) {
				my_index = 0;
				base_index = atomicAdd(bottom, count);
			}
			base_index = __shfl_sync(-1, base_index, 0);
			size_t swap_index = base_index + my_index;
			if (found_swap) {
				if (!(parts.pos(xdim, swap_index) < xmid)) {
					parts.swap(i, swap_index);
					found_swap = false;
				}
			}
		}
	}
}

size_t count_particles(particle_set parts, size_t begin, size_t end, fixed32 xpos, int xdim) {
	const auto nparts = global().opts.nparts;
	const auto nprocs = global().cuda.devices[0].multiProcessorCount;
	const auto mycount = end - begin;
	int nchunks = std::max(1, (int) (mycount * nprocs / nparts));
	size_t* counts;
	CUDA_MALLOC(counts, nchunks);
	auto stream = get_stream();
	parts.prepare_sort1(xdim, stream);
	count_kernel<<<nchunks,COUNT_BLOCK_SIZE,0,stream>>>(parts,begin,end,xpos,xdim,counts);
	CUDA_CHECK(cudaStreamSynchronize(stream));
	cleanup_stream(stream);
	size_t count = 0;
	for (int i = 0; i < nchunks; i++) {
		count += counts[i];
	}
	CUDA_FREE(counts);
	return count;
}

fixed32 find_median(particle_set parts, size_t begin, size_t end, fixed32 xmin, fixed32 xmax, int xdim) {
	int64_t half = (end - begin) / 2;
	int64_t lastmid;
	int64_t countmid = 0;
	int64_t countmax;
	fixed32 xmid;
	bool first_call = true;
	do {
		xmid = fixed32((fixed64(xmin) + fixed64(xmax)) / fixed64(2));
		lastmid = countmid;
		countmid = (int64_t) count_particles(parts, begin, end, xmid, xdim) - half;
		if (first_call) {
			countmax = (int64_t) count_particles(parts, begin, end, xmax, xdim) - half;
			first_call = false;
		}
//		printf("%li %li %e %e %e \n", countmid, half, xmin.to_float(), xmid.to_float(), xmax.to_float());
		if (countmid * countmax < 0) {
			xmin = xmid;
		} else {
			xmax = xmid;
			countmax = countmid;
		}
	} while (lastmid != countmid && countmid);
	unsigned long long* bottom;
	CUDA_MALLOC(bottom, 1);
	*bottom = 0;
	const auto nparts = global().opts.nparts;
	const auto nprocs = global().cuda.devices[0].multiProcessorCount;
	const auto mycount = end - begin;
	int nchunks = std::max(1, (int) (mycount * nprocs / nparts));
	printf("Sorting %i\n", nchunks);
	auto stream = get_stream();
	parts.prepare_sort2(stream);
	sort_kernel<<<nchunks*SORT_OCCUPANCY,SORT_BLOCK_SIZE,0,stream>>>(parts, begin, end, xmid, xdim, bottom);
	CUDA_CHECK(cudaStreamSynchronize(stream));
	cleanup_stream(stream);
	printf("Done Sorting\n");
	CUDA_FREE(bottom);
	return xmid;
}
