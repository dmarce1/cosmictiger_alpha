#include <cosmictiger/groups.hpp>
#include <cosmictiger/gravity.hpp>

#define NLISTS 2
#define OPEN 0
#define NEXT 1
#define NOLIST 2

__global__ void cuda_find_groups_kernel(bool* rc, group_param_type** params);

__managed__ group_list_sizes sizes = { 0, 0 };

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

__device__ bool cuda_find_groups(tree_ptr self);

group_list_sizes get_group_list_sizes() {
	return sizes;
}

__global__ void cuda_find_groups_kernel(bool* rc, group_param_type** params) {
	extern int __shared__ shmem_ptr[];
	groups_shmem& shmem = *((groups_shmem*) shmem_ptr);
	const auto& tid = threadIdx.x;
	const auto& bid = blockIdx.x;
	auto& param = *params[bid];
	if (tid == 0) {
		memcpy(&shmem.checks, &param.checks, sizeof(shmem.checks));
		memcpy(&shmem.opened_checks, &param.opened_checks, sizeof(shmem.checks));
		memcpy(&shmem.next_checks, &param.next_checks, sizeof(shmem.checks));
		shmem.parts = param.parts;
		shmem.link_len = param.link_len;
		shmem.depth = param.depth;

	}
	__syncwarp();
	const bool this_rc = cuda_find_groups(param.self);
	__syncwarp();
	if (tid == 0) {
		//	param.call_destructors();
		atomicMax(&sizes.opened, shmem.opened_checks.capacity());
		atomicMax(&sizes.next, shmem.next_checks.capacity());
		shmem.checks.~stack_vector<tree_ptr>();
		shmem.opened_checks.~vector<tree_ptr>();
		shmem.next_checks.~vector<tree_ptr>();
		//	param.~group_param_type();
		rc[bid] = this_rc;
	}
}

__device__ bool cuda_find_groups(tree_ptr self) {
	const auto& tid = threadIdx.x;
	extern int __shared__ shmem_ptr[];
	groups_shmem& shmem = *((groups_shmem*) shmem_ptr);
	auto& self_parts = shmem.self_parts;
	auto& other_parts = shmem.others;
	auto& other_indexes = shmem.other_indexes;
	auto& checks = shmem.checks;
	auto& parts = shmem.parts;
	auto& next_checks = shmem.next_checks;
	auto& opened_checks = shmem.opened_checks;

	auto myrange = self.get_range();
	myrange.expand(1.01f * shmem.link_len);
	const auto iamleaf = self.is_leaf();
	opened_checks.resize(0);
	array<vector<tree_ptr>*, NLISTS> lists;
	array<int, NLISTS + 1> indices;
	array<int, NLISTS> counts;

	lists[NEXT] = &next_checks;
	lists[OPEN] = &opened_checks;
	int mylist;
	int rc;
	do {
		next_checks.resize(0);
		const int cimax = ((checks.size() - 1) / warpSize + 1) * warpSize;
		for (int i = tid; i < cimax; i += warpSize) {
			indices[OPEN] = indices[NEXT] = indices[NOLIST] = 0;
			mylist = NOLIST;
			if (i < checks.size()) {
				const int intersects = myrange.intersects(checks[i].get_range());
				const int other_flag = int(bool(checks[i].last_group_flag()));
				const int use = other_flag * intersects;
				const int isleaf = checks[i].is_leaf();
				mylist = use * ((1 - isleaf) * NEXT) + (1 - use) * NOLIST;
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
			assert(2 * i + RIGHT < checks.size());
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
		const auto linklen2 = sqr(shmem.link_len);
		const int mysize = myparts.second - myparts.first;
		for (int i = tid; i < mysize; i += warpSize) {
			for (int dim = 0; dim < NDIM; dim++) {
				self_parts[dim][i] = parts.pos(dim, i + myparts.first);
			}
		}

		int found_link, index, count, other_count;
		bool found;
		rc = 0;
		for (int i = 0; i < checks.size(); i++) {
			if (checks[i] == self) {
				continue;
			}
			const auto other_pair = checks[i].get_parts();
			const int other_size = other_pair.second - other_pair.first;
			const int other_max = ((other_size - 1) / warpSize + 1) * warpSize;
			other_count = 0;
			array<fixed32, NDIM> x;
			for (int k = tid; k < other_max; k += warpSize) {
				index = 0;
				found = false;
				if (k < other_size) {
					const auto k0 = k + other_pair.first;
					for (int dim = 0; dim < NDIM; dim++) {
						x[dim] = parts.pos(dim, k0);
					}
					if (myrange.contains(x)) {
						index = 1;
						found = true;
					}
				}
				for (int P = 1; P < warpSize; P *= 2) {
					const auto tmp = __shfl_up_sync(FULL_MASK, index, P);
					if (tid >= P) {
						index += tmp;
					}
				}
				count = __shfl_sync(FULL_MASK, index, warpSize - 1);
				const auto tmp = __shfl_up_sync(FULL_MASK, index, 1);
				index = (tid > 0) ? tmp : 0;
				index += other_count;
				if (found) {
					for (int dim = 0; dim < NDIM; dim++) {
						other_parts[dim][index] = x[dim];
					}
					other_indexes[index] = k;
				}
				other_count += count;
			}
			__syncwarp();
			for (int j = tid; j < mysize; j += warpSize) {
				for (int k = 0; k != other_count; k++) {
					float dx0, dx1, dx2;
					dx0 = distance(self_parts[0][j], other_parts[0][k]);
					dx1 = distance(self_parts[1][j], other_parts[1][k]);
					dx2 = distance(self_parts[2][j], other_parts[2][k]);
					const float dist2 = fmaf(dx0, dx0, fmaf(dx1, dx1, sqr(dx2)));
					if (dist2 < linklen2) {
						const auto j0 = j + myparts.first;
						const auto k0 = other_indexes[k] + other_pair.first;
						auto& id1 = parts.group(j0);
						const auto id2 = parts.group(k0);
						const auto oldid1 = id1;
						id1 = min((id1 == NO_GROUP) ? j0 : id1, id2);
						rc += (oldid1 != id1);
					}
				}
			}
			__syncwarp();
		}
		do {
			found_link = 0;
			for (int i = 0; i < checks.size(); i++) {
				__syncwarp();
				for (int j = tid; j < mysize; j += warpSize) {
					for (int k = 0; k != mysize; k++) {
						float dx0, dx1, dx2;
						dx0 = distance(self_parts[0][j], self_parts[0][k]);
						dx1 = distance(self_parts[1][j], self_parts[1][k]);
						dx2 = distance(self_parts[2][j], self_parts[2][k]);
						const float dist2 = fmaf(dx0, dx0, fmaf(dx1, dx1, sqr(dx2)));
						if (dist2 < linklen2 && j != k) {
							const auto j0 = j + myparts.first;
							const auto k0 = k + myparts.first;
							auto& id1 = parts.group(j0);
							const auto id2 = parts.group(k0);
							const auto oldid1 = id1;
							id1 = min((id1 == NO_GROUP) ? j0 : id1, id2);
							rc += (oldid1 != id1);
						}
					}
				}
				__syncwarp();
			}
		} while (__reduce_add_sync(FULL_MASK, found_link) != 0);
		rc = __reduce_add_sync(FULL_MASK, rc) != 0;
	} else {
		if (shmem.checks.size()) {
			auto mychildren = self.get_children();
			shmem.checks.push_top();
			if (tid == 0) {
				shmem.depth++;
			}
			__syncwarp();
			const auto rc1 = cuda_find_groups(mychildren[LEFT]);
			shmem.checks.pop_top();
			__syncwarp();
			const auto rc2 = cuda_find_groups(mychildren[RIGHT]);
			if (tid == 0) {
				shmem.depth--;
			}
			__syncwarp();
			rc = rc1 || rc2;
		} else {
			rc = false;
		}
	}
	if (tid == 0) {
		self.group_flag() = rc != 0;
	}
	return rc != 0;

}
