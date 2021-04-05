#include <cosmictiger/groups.hpp>
#include <cosmictiger/gravity.hpp>

#define NLISTS 3
#define OPEN 0
#define NEXT 1
#define NOLIST 2

__global__ void cuda_find_groups_kernel(bool* rc, group_param_type** params);

std::function<std::vector<bool>()> call_cuda_find_groups(group_param_type** params, int params_size,
		cudaStream_t stream) {
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

	auto self = params.self;

	const auto myrange = self.get_range();
	const auto iamleaf = self.is_leaf();
	opened_checks.resize(0);
	array<vector<tree_ptr>*, NLISTS> lists;
	array<int, NLISTS> indices;
	array<int, NLISTS> counts;

	lists[NEXT] = &next_checks;
	lists[OPEN] = &opened_checks;
	lists[NOLIST] = nullptr;
	int mylist;
	do {
		next_checks.resize(0);
		const int cimax = ((checks.size() - 1) / warpSize + 1) * warpSize;
		for (int i = tid; i < cimax; i += warpSize) {
			indices[OPEN] = indices[NEXT] = indices[NOLIST] = 0;
			mylist = NOLIST;
			if (i < checks.size() && myrange.intersects(checks[i].get_range())) {
				if (checks[i].is_leaf()) {
					mylist = OPEN;
				} else {
					mylist = NEXT;
				}
			} else {
				mylist = NOLIST;
			}
			indices[mylist] = 1;
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
	if (tid == 0) {
		for (int i = 0; i < opened_checks.size(); i++) {
			checks.push(opened_checks[i]);
		}
	}
	__syncwarp();
	if (iamleaf) {
		const auto myparts = self.get_parts();
		const auto linklen2 = sqr(params.link_len);
		int found_link, this_found_link;
		found_link = 0;
		for (int i = 0; i < checks.size(); i++) {
			const auto other_parts = checks[i].get_parts();
			for (int j = myparts.first + tid; j < myparts.second; j += warpSize) {
				for (int k = other_parts.first; k != other_parts.second; k++) {
					float dx0, dx1, dx2;
					dx0 = distance(parts.pos(0, j), parts.pos(0, k));
					dx1 = distance(parts.pos(1, j), parts.pos(1, k));
					dx2 = distance(parts.pos(2, j), parts.pos(2, k));
					const float dist2 = fma(dx0, dx0, fma(dx1, dx1, sqr(dx2)));
					if (dist2 < linklen2 && dist2 != 0.0) {
						if (parts.group(j) == -1) {
							parts.group(j) = j;
						}
						if (parts.group(k) == -1) {
							parts.group(k) = k;
						}
						auto& id1 = parts.group(k);
						auto& id2 = parts.group(j);
						const auto shared_id = min(id1, id2);
						if (id1 != shared_id || id2 != shared_id) {
							found_link++;
						}
						id1 = id2 = shared_id;
					}
				}
			}
			__syncwarp();
		}
		return __reduce_add_sync(FULL_MASK, found_link) != 0;
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

	/*
	 *
	 * array<vector<tree_ptr>*, NLISTS> lists;
	 array<int, NLISTS> indices;
	 array<int, NLISTS> counts;
	 tree_ptr self = params.self;

	 lists[NEXT] = &next_checks;
	 lists[OPEN] = &opened_checks;
	 lists[NOLIST] = nullptr;
	 *
	 const auto myrange = self.get_range();
	 const auto iamleaf = self.is_leaf();
	 int mylist;
	 opened_checks.resize(0);
	 do {
	 next_checks.resize(0);
	 __syncwarp();
	 const int cimax = ((checks.size() - 1) / warpSize + 1) * warpSize;
	 for (int i = tid; i < cimax; i += warpSize) {
	 indices[OPEN] = indices[NEXT] = indices[NOLIST] = 0;
	 mylist = NOLIST;
	 if (i < checks.size()) {
	 const auto other_range = checks[i].get_range();
	 const int intersects = myrange.intersects(other_range);
	 const int isleaf = checks[i].is_leaf();
	 mylist = intersects * (isleaf * OPEN + (1 - isleaf) * NEXT) + (1 - intersects) * NOLIST;
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
	 indices[OPEN] += osz;
	 opened_checks.resize(osz + counts[OPEN]);
	 next_checks.resize(counts[NEXT]);
	 __syncwarp();
	 if (mylist != NOLIST) {
	 assert(indices[mylist] < lists[mylist]->size());
	 (*(lists[mylist]))[indices[mylist]] = checks[i];
	 }
	 __syncwarp();
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
	 __syncwarp();
	 } while (iamleaf && checks.size());
	 const int csz = checks.size();
	 checks.resize(csz + opened_checks.size());
	 __syncwarp();
	 for (int i = tid; i < opened_checks.size(); i += warpSize) {
	 checks[csz + i] = opened_checks[i];
	 }
	 __syncwarp();
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
	 const bool rightrc = cuda_find_groups(params_ptr);
	 if (tid == 0) {
	 params.depth--;
	 }
	 __syncwarp();
	 found_link = found_link || rightrc;
	 rc = found_link;
	 } else {

	 const auto myparts = self.get_parts();
	 __syncwarp();
	 const auto linklen2 = sqr(params.link_len);
	 bool found_link;
	 for (int i = 0; i < checks.size(); i++) {
	 const auto other_parts = checks[i].get_parts();
	 found_link = false;
	 for (int j = myparts.first + tid; j < myparts.second; j += warpSize) {
	 for (int k = other_parts.first; k < other_parts.second; k++) {
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
	 }*/
	return rc;
}

