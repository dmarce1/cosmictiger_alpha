#include <cosmictiger/groups.hpp>
#include <cosmictiger/gravity.hpp>

#define NLISTS 2
#define OPEN 0
#define NEXT 1
#define NOLIST 2

__global__ void cuda_find_groups_kernel(bool* rc, group_param_type** params);

std::function<std::vector<bool>()> call_cuda_find_groups(group_param_type** params, int params_size,
		cudaStream_t stream) {
	bool* rc;
	unified_allocator alloc;
	rc = (bool*) alloc.allocate(sizeof(bool) * params_size);
	cuda_find_groups_kernel<<<params_size,WARP_SIZE,sizeof(groups_shmem),stream>>>(rc, params);
	const int size = params_size;
	return [stream,rc, size]() {
		if( cudaStreamQuery(stream)==cudaSuccess) {
			std::vector<bool> res(size);
			std::copy(rc, rc+size, res.begin());
			unified_allocator().deallocate(rc);
			return res;
		} else {
			return std::vector<bool>();
		}
	};
}

__global__ void cuda_find_groups_kernel(bool* rc, group_param_type** params) {
	const auto& tid = threadIdx.x;
	const auto& bid = blockIdx.x;
	const bool this_rc = cuda_find_groups(params[bid]);
	if (tid == 0) {
		auto& param = *params[bid];
		param.call_destructors();
		rc[bid] = this_rc;
	}
}

__device__ bool cuda_find_groups(group_param_type* params_ptr) {
	const auto& tid = threadIdx.x;
	extern int __shared__ shmem_ptr[];
	groups_shmem& shmem = *((groups_shmem*) shmem_ptr);
	auto& self_parts = shmem.self;
	auto& other_parts = shmem.others;
	bool rc;
	group_param_type& params = *params_ptr;
	auto& checks = params.checks;
	auto& parts = params.parts;
	auto& next_checks = params.next_checks;
	auto& opened_checks = params.opened_checks;

	auto self = params.self;

	const auto myrange = self.get_range();
	const auto iamleaf = self.is_leaf();
	opened_checks.resize(0);
	array<vector<tree_ptr>*, NLISTS> lists;
	array<int, NLISTS + 1> indices;
	array<int, NLISTS> counts;

	lists[NEXT] = &next_checks;
	lists[OPEN] = &opened_checks;
	int mylist;
	do {
		next_checks.resize(0);
		const int cimax = ((checks.size() - 1) / warpSize + 1) * warpSize;
		for (int i = tid; i < cimax; i += warpSize) {
			indices[OPEN] = indices[NEXT] = indices[NOLIST] = 0;
			mylist = NOLIST;
			if (i < checks.size()) {
				const int intersects = myrange.intersects(checks[i].get_range());
				const int isleaf = checks[i].is_leaf();
				mylist = intersects * ((1 - isleaf) * NEXT) + (1 - intersects) * NOLIST;
				indices[mylist] = 1;
			}
			for (int P = 1; P < warpSize; P *= 2) {
				for (int j = 0; j < NLISTS; j++) {
					const auto tmp = __shfl_up_sync(FULL_MASK, indices[j], P);
					if (tid >= P) {
						indices[j] += tmp;
					}
				}
			}
			for (int j = 0; j < NLISTS; j++) {
				counts[j] = __shfl_sync(FULL_MASK, indices[j], warpSize - 1);
				const auto tmp = __shfl_up_sync(FULL_MASK, indices[j], 1);
				indices[j] = (tid > 0) ? tmp : 0;
			}
			const int osz = opened_checks.size();
			const int nsz = next_checks.size();
			indices[OPEN] += osz;
			indices[NEXT] += nsz;
			__syncwarp();
			opened_checks.resize(osz + counts[OPEN]);
			next_checks.resize(nsz + counts[NEXT]);
			__syncwarp();
			if (mylist != NOLIST) {
				assert(indices[mylist] < lists[mylist]->size());
				(*(lists[mylist]))[indices[mylist]] = checks[i];
			}
		}
		__syncwarp();
		checks.resize(NCHILD * next_checks.size());
		__syncwarp();
		for (int i = tid; i < next_checks.size(); i += warpSize) {
			const auto children = next_checks[i].get_children();
			assert(2*i+RIGHT < checks.size());
			checks[NCHILD * i + LEFT] = children[LEFT];
			checks[NCHILD * i + RIGHT] = children[RIGHT];
		}
	} while (iamleaf && checks.size());
	__syncwarp();
	const int csz = checks.size();
	checks.resize(csz + opened_checks.size());
	__syncwarp();
	for (int i = tid; i < opened_checks.size(); i += warpSize) {
		checks[i + csz] = opened_checks[i];
	}
	__syncwarp();
	if (iamleaf) {
		const auto myparts = self.get_parts();
		const auto linklen2 = sqr(params.link_len);

		for (int i = tid; i < (myparts.second - myparts.first); i += warpSize) {
			for (int dim = 0; dim < NDIM; dim++) {
				self_parts[dim][i] = parts.pos(dim, i + myparts.first);
			}
		}

		int found_link, iters;
		iters = 0;
		do {
			found_link = 0;
			for (int i = 0; i < checks.size(); i++) {
				const auto other_pair = checks[i].get_parts();
				for (int k = tid; k < (other_pair.second - other_pair.first); k += warpSize) {
					for (int dim = 0; dim < NDIM; dim++) {
						other_parts[dim][k] = parts.pos(dim, k + other_pair.first);
					}
				}
				__syncwarp();
				for (int k = 0; k != other_pair.second - other_pair.first; k++) {
					for (int j = tid; j < myparts.second - myparts.first; j += warpSize) {

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
		return iters > 1;
	} else {
		auto mychildren = self.get_children();
		params.checks.push_top();
		if (tid == 0) {
			params.self = mychildren[LEFT];
			params.depth++;
		}
		__syncwarp();
		const auto rc1 = cuda_find_groups(params_ptr);
		params.checks.pop_top();
		if (tid == 0) {
			params.self = mychildren[RIGHT];
		}
		__syncwarp();
		const auto rc2 = cuda_find_groups(params_ptr);
		if (tid == 0) {
			params.depth--;
		}
		__syncwarp();
		return rc1 || rc2;
	}

}

