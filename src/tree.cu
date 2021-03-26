#include <cosmictiger/cuda.hpp>
#include <stack>

#define TREECU
#include <cosmictiger/tree.hpp>
#include <cosmictiger/interactions.hpp>
#include <functional>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/kick_return.hpp>

//CUDA_KERNEL cuda_kick()

#define CC_CP_DIRECT 0
#define CC_CP_EWALD 1
#define PC_PP_DIRECT 2
#define PC_PP_EWALD 3
#define N_INTERACTION_TYPES 4

CUDA_DEVICE particle_set *parts;

__managed__ list_sizes_t list_sizes { 0, 0, 0, 0, 0 };

#define MI 0
#define PI 1
#define CI 2
#define OI 3

CUDA_DEVICE void cuda_kick(kick_params_type * params_ptr) {
	kick_params_type &params = *params_ptr;
	__shared__
	extern int shmem_ptr[];
	cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
	tree_ptr tptr = params.tptr;
	tree& me = *((tree*) params.tptr);
	if (!me.active_parts && !params.full_eval) {
		return;
	}
	const int &tid = threadIdx.x;
	auto &F = params.F;
	auto &phi = params.Phi;
	auto &L = params.L[params.depth];
	int list_index;
	if (tid == 0) {
		const auto &Lpos = params.Lpos[params.depth];
		array<float, NDIM> dx;
		for (int dim = 0; dim < NDIM; dim++) {
			const auto x1 = me.pos[dim];
			const auto x2 = Lpos[dim];
			dx[dim] = distance(x1, x2);
		}
		shift_expansion(L, dx, params.full_eval);
	}
	int flops = 0;
	int interacts = 0;
	if (((tree*) tptr)->children[0].ptr == 0) {
		for (int k = tid; k < MAX_BUCKET_SIZE; k += KICK_BLOCK_SIZE) {
			for (int dim = 0; dim < NDIM; dim++) {
				F[dim][k] = 0.f;
			}
			phi[k] = -PHI0;
		}
	}
	const auto& myparts = ((tree*) params.tptr)->parts;
	array<int, NITERS> count;

	const float theta = params.theta;
	const float theta2 = params.theta * params.theta;
	array<vector<tree_ptr>*, NITERS> lists;
	auto &multis = params.multi_interactions;
	auto &parti = params.part_interactions;
	auto &next_checks = params.next_checks;
	auto &opened_checks = params.opened_checks;
	lists[MI] = &multis;
	lists[PI] = &parti;
	lists[CI] = &next_checks;
	lists[OI] = &opened_checks;
	int my_index[NITERS];
	int index_counts[NITERS];
	const float myradius1 = me.radius; // + h;
	const float myradius2 = SINK_BIAS * myradius1;
	const auto mypos = me.pos;
	const bool iamleaf = me.children[0].ptr == 0;
	int ninteractions = iamleaf ? 3 : 2;
	for (int type = 0; type < ninteractions; type++) {
		const bool ewald_dist = type == PC_PP_EWALD || type == CC_CP_EWALD;
		auto& checks = ewald_dist ? params.echecks : params.dchecks;
		const bool direct = type == PC_PP_EWALD || type == PC_PP_DIRECT;
		for (int i = 0; i < NITERS; i++) {
			count[i] = 0;
		}
		for (int i = 0; i < NITERS; i++) {
			lists[i]->resize(0, vectorPOD);
		}
		__syncwarp();
		int check_count;
		array<int, NITERS> tmp;
		const float h = params.hsoft;
		const float th = params.theta * fmaxf(h, MIN_DX);
		do {
			check_count = checks.size();
			if (check_count) {
				const int cimax = ((check_count - 1) / KICK_BLOCK_SIZE + 1) * KICK_BLOCK_SIZE;
				for (int ci = tid; ci < cimax; ci += KICK_BLOCK_SIZE) {
					for (int i = 0; i < NITERS; i++) {
						my_index[i] = 0;
					}
					if (ci < check_count) {
						const auto check = checks[ci];
						const auto other_pos = ((const tree*) check)->pos;
						const float other_radius = ((const tree*) check)->radius;
						const float other_nparts = ((const tree*) check)->parts.second - ((const tree*) check)->parts.first;
						;
						array<float, NDIM> dist;
						for (int dim = 0; dim < NDIM; dim++) {                         // 3
							dist[dim] = distance(other_pos[dim], mypos[dim]);
						}
						float d2 = fmaf(dist[0], dist[0], fmaf(dist[1], dist[1], sqr(dist[2]))); // 5
						if (ewald_dist) {
							d2 = fmaxf(d2, EWALD_MIN_DIST2); // 5
						}
						const float R1 = sqr(other_radius + myradius2 + th);                 // 2
						const float R2 = sqr(other_radius * theta + myradius2 + th); // 3
						const float R3 = sqr(other_radius + myradius1 * theta + th); // 3
						const float theta2d2 = theta2 * d2; // 1
						const bool far1 = R1 < theta2d2;                 // 1
						const bool far2 = R2 < theta2d2;                 // 1
						const bool far3 = R3 < theta2d2;                 // 1
						const bool isleaf = (((const tree*) check)->children[0].ptr == 0);
						interacts++;
						flops += 27;
						const bool mi = far1 || (direct && far3 && other_nparts >= MIN_PC_PARTS);
						const bool pi = (far2 || direct) && isleaf;
						list_index = int(mi) * MI
								+ (1 - int(mi)) * (int(pi) * PI + (1 - int(pi)) * (int(isleaf) * OI + (1 - int(isleaf)) * CI));
						my_index[list_index] = 1;
					}
					for (int P = 1; P < KICK_BLOCK_SIZE; P *= 2) {
						for (int i = 0; i < NITERS; i++) {
							tmp[i] = __shfl_up_sync(0xFFFFFFFF, my_index[i], P);
							if (tid >= P) {
								my_index[i] += tmp[i];
							}
						}
					}
					for (int i = 0; i < NITERS; i++) {
						index_counts[i] = __shfl_sync(0xFFFFFFFF, my_index[i], KICK_BLOCK_SIZE - 1);
						tmp[i] = __shfl_up_sync(0xFFFFFFFF, my_index[i], 1);
						if (tid >= 1) {
							my_index[i] = tmp[i];
						} else {
							my_index[i] = 0;
						}
					}
					for (int i = 0; i < NITERS; i++) {
						lists[i]->resize(count[i] + index_counts[i], vectorPOD);
					}
					__syncwarp();
					if (ci < check_count) {
						const auto &check = checks[ci];
						assert(count[list_index] + my_index[list_index] >= 0);
						(*lists[list_index])[count[list_index] + my_index[list_index]] = check;
					}
					for (int i = 0; i < NITERS; i++) {
						count[i] += index_counts[i];
					}
				}
				__syncwarp();
				auto& countCI = count[CI];
				auto& countOI = count[OI];
				check_count = 2 * countCI + countOI;
				checks.resize(check_count);
				__syncwarp();
				for (int i = tid; i < countCI; i += KICK_BLOCK_SIZE) {
					const auto base = 2 * i;
					const auto children = next_checks[i].get_children();
					for (int j = 0; j < NCHILD; j++) {
						checks[base + j] = children[j];
					}
				}
				const auto base = 2 * countCI;
				for (int i = tid; i < countOI; i += KICK_BLOCK_SIZE) {
					checks[base + i] = opened_checks[i];
				}
				__syncwarp();
				countCI = 0;
				countOI = 0;
				next_checks.resize(0, vectorPOD);
				opened_checks.resize(0), vectorPOD;
			}
		} while (direct && check_count);
		auto &tmp_parti = params.tmp;
		if (type == PC_PP_DIRECT) {
			auto &sinks = shmem.sink;
			const int pmax = max(((parti.size() - 1) / KICK_BLOCK_SIZE + 1) * KICK_BLOCK_SIZE, 0);
			for (int k = tid; k < myparts.second - myparts.first; k += KICK_BLOCK_SIZE) {
				const auto index = k + myparts.first;
				if (parts->rung(index) >= params.rung || params.full_eval) {
					for (int dim = 0; dim < NDIM; dim++) {
						sinks[dim][k] = parts->pos(dim, index);
					}
				}
			}
			tmp_parti.resize(0, vectorPOD);
			__syncwarp();
			for (int j = tid; j < pmax; j += KICK_BLOCK_SIZE) {
				my_index[0] = 0;
				my_index[1] = 0;
				list_index = -1;
				if (j < parti.size()) {
					const auto& other = *((tree*) parti[j]);
					const auto other_nparts = other.parts.second - other.parts.first;
					const float hfac = params.theta * fmaxf(params.hsoft, MIN_DX);
					bool res = false;
					const int sz = myparts.second - myparts.first;
					if (other_nparts < MIN_PC_PARTS) {
						res = true;
					} else {
						for (int k = 0; k < sz; k++) {
							const auto this_rung = parts->rung(k + myparts.first);
							if (this_rung >= params.rung || params.full_eval) {
								float dx0 = distance(other.pos[0], sinks[0][k]);
								float dy0 = distance(other.pos[1], sinks[1][k]);
								float dz0 = distance(other.pos[2], sinks[2][k]);
								float d2 = fma(dx0, dx0, fma(dy0, dy0, sqr(dz0)));
								res = sqr(other.radius + hfac) > d2 * theta2;
								flops += 15;
								if (res) {
									break;
								}
							}
						}
					}
					list_index = res ? PI : MI;
					my_index[list_index] = 1;
				}
				for (int P = 1; P < KICK_BLOCK_SIZE; P *= 2) {
					for (int i = 0; i < 2; i++) {
						tmp[i] = __shfl_up_sync(0xFFFFFFFF, my_index[i], P);
						if (tid >= P) {
							my_index[i] += tmp[i];
						}
					}
				}
				for (int i = 0; i < 2; i++) {
					index_counts[i] = __shfl_sync(0xFFFFFFFF, my_index[i], KICK_BLOCK_SIZE - 1);
					tmp[i] = __shfl_up_sync(0xFFFFFFFF, my_index[i], 1);
					if (tid >= 1) {
						my_index[i] = tmp[i];
					} else {
						my_index[i] = 0;
					}
				}
				const auto part_cnt = tmp_parti.size();
				const auto mult_cnt = multis.size();
				tmp_parti.resize(part_cnt + index_counts[PI], vectorPOD);
				multis.resize(mult_cnt + index_counts[MI]), vectorPOD;
				__syncwarp();
				if (j < parti.size()) {
					(list_index == PI ? tmp_parti[part_cnt + my_index[PI]] : multis[mult_cnt + my_index[MI]]) = parti[j];
				}
			}
			parti.swap(tmp_parti);
			__syncwarp();
		}
		int nactive;
		switch (type) {
		case PC_PP_DIRECT:
			nactive = compress_sinks(params_ptr);
			cuda_pc_interactions(params_ptr, nactive);
			cuda_pp_interactions(params_ptr, nactive);
			break;
		case CC_CP_DIRECT:
			cuda_cc_interactions(params_ptr, DIRECT);
			cuda_cp_interactions(params_ptr);
			break;
		case CC_CP_EWALD:
			cuda_cc_interactions(params_ptr, EWALD);
			break;
		}
	}
	if (!(((tree*) tptr)->children[0].ptr == 0)) {
		tree* left = ((tree*) tptr)->children[LEFT];
		tree* right = ((tree*) tptr)->children[RIGHT];
		if (tid == 0) {
			params.depth++;
			params.L[params.depth] = L;
			params.Lpos[params.depth] = me.pos;
		}
		if (left->active_parts && right->active_parts) {
			params.dchecks.push_top();
			params.echecks.push_top();
			if (tid == 0) {
				params.tptr = ((tree*) tptr)->children[LEFT];
			}
			__syncwarp();
			cuda_kick(params_ptr);
			if (tid == 0) {
				params.L[params.depth] = L;
				params.tptr = ((tree*) tptr)->children[RIGHT];
			}
			params.dchecks.pop_top();
			params.echecks.pop_top();
			__syncwarp();
			cuda_kick(params_ptr);
		} else if (left->active_parts) {
			if (tid == 0) {
				params.tptr = ((tree*) tptr)->children[LEFT];
			}
			__syncwarp();
			cuda_kick(params_ptr);
		} else if (right->active_parts) {
			if (tid == 0) {
				params.tptr = ((tree*) tptr)->children[RIGHT];
			}
			__syncwarp();
			cuda_kick(params_ptr);
		}
		if (tid == 0) {
			params.depth--;
		}
		//   printf( "%li\n", rc.flops);
	} else {
		const float invlog2 = 1.0f / logf(2);
		for (int k = tid; k < myparts.second - myparts.first; k += KICK_BLOCK_SIZE) {
			const auto this_rung = parts->rung(k + myparts.first);
			if (this_rung >= params.rung || params.full_eval) {
				array<float, NDIM> g;
				float this_phi;
				array<float, NDIM> dx;
				for (int dim = 0; dim < NDIM; dim++) {
					const auto x2 = me.pos[dim];
					const auto x1 = parts->pos(dim, k + myparts.first);
					dx[dim] = distance(x1, x2);
				}
				shift_expansion(L, g, this_phi, dx, params.full_eval);
				for (int dim = 0; dim < NDIM; dim++) {
					F[dim][k] += g[dim];
				}
				phi[k] += this_phi;
				for (int dim = 0; dim < NDIM; dim++) {
					F[dim][k] *= params.G * params.M;
				}
				phi[k] *= params.G * params.M;
#ifdef TEST_FORCE
				for (int dim = 0; dim < NDIM; dim++) {
					parts->force(dim, k + myparts.first) = F[dim][k];
				}
				parts->pot(k + myparts.first) = phi[k];
#endif
				if (this_rung >= params.rung) {
					float dt = params.t0 / (size_t(1) << this_rung);
					if (!params.first) {
						parts->vel(k + myparts.first).p.x += 0.5 * dt * F[0][k];
						parts->vel(k + myparts.first).p.y += 0.5 * dt * F[1][k];
						parts->vel(k + myparts.first).p.z += 0.5 * dt * F[2][k];
					}
					float fmag = 0.0;
					for (int dim = 0; dim < NDIM; dim++) {
						fmag += sqr(F[dim][k]);
					}
					fmag = sqrtf(fmag);
					//   printf( "%e\n", fmag);
					assert(fmag > 0.0);
					dt = fminf(params.eta * sqrtf(params.scale * params.hsoft / fmag), params.t0);
					int new_rung = fmaxf(fmaxf(fmaxf(ceil(logf(params.t0 / dt) * invlog2), this_rung - 1), params.rung),
					MIN_RUNG);
					dt = params.t0 / (size_t(1) << new_rung);
					parts->vel(k + myparts.first).p.x += 0.5 * dt * F[0][k];
					parts->vel(k + myparts.first).p.y += 0.5 * dt * F[1][k];
					parts->vel(k + myparts.first).p.z += 0.5 * dt * F[2][k];
					parts->set_rung(new_rung, k + myparts.first);
				}
				if (params.full_eval) {
					kick_return_update_pot_gpu(phi[k], F[0][k], F[1][k], F[2][k]);
				}
			}
			kick_return_update_rung_gpu(parts->rung(k + myparts.first));
		}
	}
	if (params.full_eval) {
		kick_return_update_interactions_gpu(KR_OP, interacts, flops);
	}
}

CUDA_KERNEL cuda_set_kick_params_kernel(particle_set *p) {
	if (threadIdx.x == 0) {
		parts = p;
		expansion_init();
	}
}

void tree::show_timings() {
}

void tree::cuda_set_kick_params(particle_set *p) {
	cuda_set_kick_params_kernel<<<1,1>>>(p);
	CUDA_CHECK(cudaDeviceSynchronize());
	expansion_init_cpu();
}

CUDA_KERNEL cuda_kick_kernel(kick_params_type *params) {
	const int &bid = blockIdx.x;
	params[bid].particles = parts;
	cuda_kick(params + bid);
	if (threadIdx.x == 0) {
		atomicMax(&list_sizes.tmp, params[bid].tmp.capacity());
		atomicMax(&list_sizes.multi, params[bid].multi_interactions.capacity());
		atomicMax(&list_sizes.part, params[bid].part_interactions.capacity());
		atomicMax(&list_sizes.next, params[bid].next_checks.capacity());
		atomicMax(&list_sizes.open, params[bid].opened_checks.capacity());
		params[bid].kick_params_type::~kick_params_type();
	}

}

list_sizes_t get_list_sizes() {
	return list_sizes;
}

void reset_list_sizes() {
	list_sizes.tmp = list_sizes.multi = list_sizes.part = list_sizes.open = list_sizes.next = 0;
}

thread_local static std::stack<cudaStream_t> streams;

cudaStream_t get_stream() {
	if (streams.empty()) {
		cudaStream_t stream;
		CUDA_CHECK(cudaStreamCreate(&stream));
		streams.push(stream);
	}
	auto stream = streams.top();
	streams.pop();
	return stream;
}

void cleanup_stream(cudaStream_t s) {
	streams.push(s);
}

void cuda_execute_kick_kernel(kick_params_type *params, int grid_size, cudaStream_t stream) {
	const size_t shmemsize = sizeof(cuda_kick_shmem);
	/***************************************************************************************************************************************************/
	/**/cuda_kick_kernel<<<grid_size, KICK_BLOCK_SIZE, shmemsize, stream>>>(params);/**/
	/***************************************************************************************************************************************************/

}

