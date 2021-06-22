#include <cosmictiger/rockstar.hpp>
#include <cosmictiger/vector.hpp>
#include <cosmictiger/array.hpp>

__device__ int reduce_index(int& index) {
	const auto tid = threadIdx.x;
	for (int P = 1; P < warpSize; P *= 2) {
		const auto tmp = __shfl_up_sync(0xFFFFFFFF, index, P);
		if (tid >= P) {
			index += tmp;
		}
	}
	int count = __shfl_sync(0xFFFFFFFF, index, warpSize - 1);
	auto tmp = __shfl_up_sync(0xFFFFFFFF, index, 1);
	if (tid >= 1) {
		index = tmp;
	} else {
		index = 0;
	}
	return count;
}

__global__ void rockstar_bh_kernel(halo_part* parts, halo_tree* trees, int nparts, float h) {

	const auto tid = threadIdx.x;
	const auto bid = blockIdx.x;
	const auto gsz = blockDim.x;
	const float hinv = 1.0f / h;
	const float h2 = sqr(h);
	const float thetainv2 = sqr(1.f / 0.7f);

	const int begin = size_t(bid) * size_t(nparts) / size_t(gsz);
	const int end = size_t(bid + 1) * size_t(nparts) / size_t(gsz);

	vector<int> current_list;
	vector<int> next_list;
	vector<int> mono_list;

	for (int pi = begin; pi < end; pi++) {
		current_list.resize(1);
		current_list[0] = 0;
		next_list.resize(0);
		mono_list.resize(0);
		__syncthreads();
		const auto& x = parts[pi].x;
		parts[pi].phi = -PHI0 * hinv;
		while (current_list.size()) {
			for (int ci = tid; ci < current_list.size(); ci += warpSize) {
				const auto& tree_node = trees[current_list[ci]];
				const auto dx = tree_node.x[0] - x[0];
				const auto dy = tree_node.x[1] - x[1];
				const auto dz = tree_node.x[2] - x[2];
				const auto d2 = fmaf(dx, dx, fmaf(dy, dy, sqr(dz)));
				int near, far, index, count;
				if ((sqr(tree_node.radius) * thetainv2 < d2) || (tree_node.children[0] == -1)) {
					far = 1;
					near = 0;
				} else {
					far = 0;
					near = 1;
				}
				index = near;
				count = reduce_index(index);
				auto offset = next_list.size();
				next_list.resize(NCHILD * count + offset, vectorPOD);
				__syncthreads();
				if (near) {
					next_list[offset + NCHILD * index + LEFT] = tree_node.children[LEFT];
					next_list[offset + NCHILD * index + RIGHT] = tree_node.children[RIGHT];
				}
				index = far;
				count = reduce_index(index);
				offset = mono_list.size();
				mono_list.resize(count + offset, vectorPOD);
				__syncthreads();
				if (near) {
					mono_list[offset + index] = current_list[ci];
				}
			}
			current_list.swap(next_list);
			float phi = 0.f;
			for (int j = tid; j < mono_list.size(); j += warpSize) {
				const auto& tree_node = trees[mono_list[j]];
				const float dx = x[0] - tree_node.x[0];
				const float dy = x[1] - tree_node.x[1];
				const float dz = x[2] - tree_node.x[2];
				const float r2 = fmaf(dx, dx, fmaf(dy, dy, sqr(dz)));
				float rinv;
				if (r2 >= h2) {
					rinv = rsqrtf(r2);
				} else {
					const float q = sqrtf(r2) * hinv;
					const float q2 = sqr(q);
					rinv = -5.0f / 16.0f;
					rinv = fmaf(rinv, q2, 21.0f / 16.0f);
					rinv = fmaf(rinv, q2, -35.0f / 16.0f);
					rinv = fmaf(rinv, q2, 35.0f / 16.0f);
					rinv *= hinv;
				}
				phi -= rinv;
			}
			for (int P = warpSize / 2; P >= 1; P /= 2) {
				phi += __shfl_down_sync(0xffffffff, phi, P);
			}
			if (tid == 0) {
				parts[pi].phi += phi;
			}
		}
	}
}
