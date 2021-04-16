#include <cosmictiger/groups.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/groups.hpp>

#define NLISTS 2
#define OPEN 0
#define NEXT 1
#define NOLIST 2

#define WARPSIZE 32

__global__ void cuda_find_groups_kernel_phase2(group_param_type* params_ptr, const tree_ptr* leaves, int Nleaves);

__managed__ int counter;

bool call_cuda_find_groups_phase2(group_param_type* params, const vector<tree_ptr>& leaves) {
	size_t dummy, total_mem;
	CUDA_CHECK(cudaMemGetInfo(&dummy, &total_mem));
	total_mem /= 8;
	size_t used_mem = (sizeof(group_t) + sizeof(fixed32) * NDIM) * params->parts.size();
	int num_rounds = std::max(1, (int) (used_mem / total_mem));
	int numBlocks;
	CUDA_CHECK(
			cudaOccupancyMaxActiveBlocksPerMultiprocessor ( &numBlocks, cuda_find_groups_kernel_phase2, WARPSIZE, sizeof(groups_shmem) ));
	numBlocks *= global().cuda.devices[0].multiProcessorCount;
	counter = 0;
	for (int round = 0; round < num_rounds; round++) {
		const int start = round * leaves.size() / num_rounds;
		const int stop = (round + 1) * leaves.size() / num_rounds;
		cuda_find_groups_kernel_phase2<<<numBlocks,WARPSIZE,sizeof(groups_shmem)>>>(params,leaves.data()+ start,stop-start);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	return counter;
}

__global__ void cuda_find_groups_kernel_phase2(group_param_type* params_ptr, const tree_ptr* leaves, int Nleaves) {
	extern int __shared__ shmem_ptr[];
	groups_shmem& shmem = *((groups_shmem*) shmem_ptr);
	group_param_type& params = *params_ptr;
	auto& self_parts = shmem.self;
	auto& other_parts = shmem.others;
	auto& parts = params.parts;
	const auto& tid = threadIdx.x;
	const auto& bid = blockIdx.x;
	const auto& gsz = gridDim.x;
	const auto start = bid * Nleaves / gsz;
	const auto stop = (bid + 1) * Nleaves / gsz;
	for (int l = start; l < stop; l++) {
		auto self = leaves[l];
		auto& checks = self.get_neighbors_ref();
		auto myparts = self.get_parts();
		int mysize = myparts.second - myparts.first;
		auto linklen2 = sqr(params.link_len);
		int found_link, iters;
		iters = 0;
		for (int k = tid; k < mysize; k += warpSize) {
			for (int dim = 0; dim < NDIM; dim++) {
				self_parts[dim][k] = parts.pos(dim, k + myparts.first);
			}
		}
		do {
			found_link = 0;

			for (auto i = checks.begin(); i != checks.end(); ++i) {
				const auto other_pair = (*i).get_parts();
				const int other_size = other_pair.second - other_pair.first;
				for (int k = tid; k < other_size; k += warpSize) {
					for (int dim = 0; dim < NDIM; dim++) {
						other_parts[dim][k] = parts.pos(dim, k + other_pair.first);
					}
				}
				__syncwarp();
				for (int j = tid; j < mysize; j += warpSize) {
					for (int k = 0; k != other_size; k++) {
						float dx0, dx1, dx2;
						dx0 = distance(self_parts[0][j], other_parts[0][k]);
						dx1 = distance(self_parts[1][j], other_parts[1][k]);
						dx2 = distance(self_parts[2][j], other_parts[2][k]);
						const float dist2 = fma(dx0, dx0, fma(dx1, dx1, sqr(dx2)));
						if (dist2 < linklen2 && dist2 != 0.0) {
							const auto j0 = j + myparts.first;
							const auto k0 = k + other_pair.first;
							auto& id1 = parts.group(j0);
							auto& id2 = parts.group(k0);
							if (atomicCAS(&id1, NO_GROUP, j0) == NO_GROUP) {
								found_link++;
							}
							if (atomicCAS(&id2, NO_GROUP, j0) == NO_GROUP) {
								found_link++;
							}
							if (atomicMin(&id1, id2) != id2) {
								found_link++;
							}
							if (atomicMin(&id2, id1) != id1) {
								found_link++;
							}
						}
					}
				}
				__syncwarp();
			}
			iters++;
		} while (__reduce_add_sync(FULL_MASK, found_link) != 0);
		if (iters > 1) {
			atomicAdd(&counter, 1);
		}
	}
}
