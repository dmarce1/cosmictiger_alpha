#include <cosmictiger/groups.hpp>
#include <cosmictiger/gravity.hpp>

#define NLISTS 2
#define OPEN 0
#define NEXT 1
#define NOLIST 2

__global__ void cuda_find_groups_kernel(bool* rc, group_param_type** params);

std::function<std::vector<bool>()> call_cuda_find_groups(group_param_type** params, int params_size, cudaStream_t stream) {
	bool* rc;
	unified_allocator alloc;
	rc = (bool*) alloc.allocate(sizeof(bool) * params_size);
	cuda_find_groups_kernel<<<params_size,WARP_SIZE,0,stream>>>(rc, params);
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
	bool rc;
		group_param_type& params = *params_ptr;
	auto& checks = params.checks;
	auto& parts = params.parts;
	auto& next_checks = params.next_checks;
	auto& opened_checks = params.opened_checks;
	array<vector<tree_ptr>*, NLISTS> lists;
	array<int, NLISTS + 1> indices;
	array<int, NLISTS + 1> counts;
	array<int, NLISTS + 1> tmp;
	tree_ptr self = params.self;

	lists[OPEN] = &next_checks;
	lists[NEXT] = &opened_checks;

	const auto myrange = self.get_range();
	const auto iamleaf = self.is_leaf();
	int mylist;
	opened_checks.resize(0);
	do {
		next_checks.resize(0);
		__syncwarp();
		indices[OPEN] = 0;
		indices[NEXT] = 0;
		for (int i = tid; i < checks.size(); i += warpSize) {
			const auto other_range = checks[i].get_range();
			const int intersects = myrange.intersects(other_range);
			const int isleaf = checks[i].is_leaf();
			mylist = intersects * (isleaf * OPEN + (1 - isleaf) * NEXT) + (1 - intersects) * NOLIST;
			indices[mylist] = 1;
			for (int P = 1; P < warpSize; P *= 2) {
				for (int i = 0; i < NLISTS; i++) {
					tmp[i] = __shfl_up_sync(0xFFFFFFFF, indices[i], P);
					if (tid >= P) {
						indices[i] += tmp[i];
					}
				}
			}
			for (int j = 0; j < NLISTS; j++) {
				counts[j] = __shfl_sync(0xFFFFFFFF, indices[j], warpSize - 1);
				tmp[j] = __shfl_up_sync(0xFFFFFFFF, indices[j], 1);
				indices[j] = (tid > 0) ? tmp[j] : 0;
			}
			const int osz = opened_checks.size();
			indices[OPEN] += osz;
			opened_checks.resize(osz + counts[OPEN]);
			next_checks.resize(NCHILD * counts[NEXT]);
			__syncwarp();
			if (mylist != NOLIST) {
				(*(lists[mylist]))[indices[mylist]] = checks[i];
			}
		}
		checks.resize(2 * next_checks.size());
		__syncwarp();
		for (int i = tid; i < next_checks.size(); i += warpSize) {
			const auto children = next_checks[i].get_children();
			checks[2 * i + LEFT] = children[LEFT];
			checks[2 * i + RIGHT] = children[RIGHT];
		}
	} while (iamleaf && checks.size());
	const int csz = checks.size();
	checks.resize(csz + opened_checks.size());
	__syncwarp();
	for (int i = tid; i < opened_checks.size(); i += warpSize) {
		checks[csz + i] = opened_checks[i];
	}
	if (!iamleaf) {
		auto mychildren = self.get_children();
		params.checks.push_top();
		if (tid == 0) {
			params.self = mychildren[LEFT];
			params.depth++;
		}
		__syncwarp();
		bool found_link = cuda_find_groups(params_ptr);
		params.checks.pop_top();
		if (tid == 0) {
			params.self = mychildren[RIGHT];
		}
		__syncwarp();
		if (tid == 0) {
			params.depth--;
		}
		__syncwarp();
		const bool rightrc = cuda_find_groups(params_ptr);
		found_link = found_link || rightrc;
		rc = found_link;
	} else {

		const auto myparts = self.get_parts();
		if (params.first_round) {
			for (auto i = myparts.first + tid; i != myparts.second; i += warpSize) {
				parts.group(i) = -1;
			}
		}
		__syncwarp();
		const auto linklen2 = sqr(params.link_len);
		bool found_link = true;
		for (int i = 0; i < checks.size(); i++) {
			const auto other_parts = checks[i].get_parts();
			found_link = false;
			for (int j = myparts.first + tid; j != myparts.second; j += warpSize) {
				for (int k = other_parts.first; k != other_parts.second; k++) {
					float dx0, dx1, dx2;
					dx0 = distance(parts.pos(0, j), parts.pos(0, k));
					dx1 = distance(parts.pos(1, j), parts.pos(1, k));
					dx2 = distance(parts.pos(2, j), parts.pos(2, k));
					const float dist2 = fma(dx0, dx0, fma(dx1, dx1, sqr(dx2)));
					if (dist2 < linklen2 && dist2 != 0.0) {
						const size_t min_index = min(j, k);
						const size_t max_index = max(j, k);
						if (parts.group(min_index) == -1) {
							parts.group(min_index) = min_index;
						}
						if (parts.group(max_index) != parts.group(min_index)) {
							found_link = true;
							parts.group(max_index) = parts.group(min_index);
						}
					}
				}
			}
		}
		rc = __reduce_add_sync(FULL_MASK, found_link);
	}
	return rc;
}

