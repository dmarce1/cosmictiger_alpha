#include <cosmictiger/cuda.hpp>
#include <stack>

#define TREECU
#include <cosmictiger/tree.hpp>
#include <cosmictiger/interactions.hpp>
#include <functional>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/kick_return.hpp>
#include <cosmictiger/tree_database.hpp>

//CUDA_KERNEL cuda_kick()

#define CC_CP_DIRECT 0
#define CC_CP_EWALD 1
#define PC_PP_DIRECT 2
#define PC_PP_EWALD 3
#define N_INTERACTION_TYPES 4

CUDA_DEVICE particle_set *parts;

__managed__ list_sizes_t list_sizes { 0, 0, 0, 0 };

#define MI 0
#define PI 1
#define CI 2
#define OI 3

__constant__ kick_constants constant;

void cuda_set_kick_constants(kick_constants consts) {
	consts.theta2 = sqr(consts.theta);
	consts.invlog2 = 1.0f / logf(2);
	consts.GM = consts.G * consts.M;
	consts.tfactor = consts.eta * sqrtf(consts.scale * consts.h);
	consts.logt0 = logf(consts.t0);
	consts.halft0 = 0.5f * consts.t0;
	consts.minrung = fmaxf(MIN_RUNG, consts.rung);
	consts.h2 = sqr(consts.h);
	consts.hinv = 1.f / consts.h;
	consts.th = consts.h * consts.theta;
	CUDA_CHECK(cudaMemcpyToSymbol(constant, &consts, sizeof(kick_constants)));
}

__constant__ float rung_dt[MAX_RUNG] = { 1.0 / (1 << 0), 1.0 / (1 << 1), 1.0 / (1 << 2), 1.0 / (1 << 3), 1.0 / (1 << 4),
		1.0 / (1 << 5), 1.0 / (1 << 6), 1.0 / (1 << 7), 1.0 / (1 << 8), 1.0 / (1 << 9), 1.0 / (1 << 10), 1.0 / (1 << 11),
		1.0 / (1 << 12), 1.0 / (1 << 13), 1.0 / (1 << 14), 1.0 / (1 << 15), 1.0 / (1 << 16), 1.0 / (1 << 17), 1.0
				/ (1 << 18), 1.0 / (1 << 19), 1.0 / (1 << 20), 1.0 / (1 << 21), 1.0 / (1 << 22), 1.0 / (1 << 23) };

CUDA_DEVICE void cuda_kick(kick_params_type * params_ptr) {
	kick_params_type &params = *params_ptr;
	__shared__
	extern int shmem_ptr[];
	cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
	particle_set& parts = *(particle_set*) constant.particles;
	tree_ptr tptr = shmem.self;
	const int &tid = threadIdx.x;
	auto &F = params.F;
	auto &phi = params.Phi;
	auto &L = params.L[shmem.depth];
	int list_index;
	const auto mypos = shmem.self.get_pos();
	const auto myradius = shmem.self.get_radius();
	const auto &Lpos = params.Lpos[shmem.depth];
	const bool iamleaf = tptr.is_leaf();
	array<float, NDIM> dx;
	for (int dim = 0; dim < NDIM; dim++) {
		dx[dim] = distance(mypos[dim], Lpos[dim]);
	}
	cuda_shift_expansion(L, dx, constant.full_eval);
	int flops = 0;
	int interacts = 0;
	if (iamleaf) {
		for (int k = tid; k < MAX_BUCKET_SIZE; k += KICK_BLOCK_SIZE) {
			for (int dim = 0; dim < NDIM; dim++) {
				F[dim][k] = 0.f;
			}
			phi[k] = -PHI0;
		}
	}
	const auto myparts = shmem.self.get_parts();
	array<int, NITERS> count;

	const float& theta = constant.theta;
	const float& theta2 = constant.theta2;
	array<vector<tree_ptr>*, NITERS> lists;
	auto &multis = shmem.multi_interactions;
	auto &parti = shmem.part_interactions;
	auto &next_checks = shmem.next_checks;
	auto &opened_checks = shmem.opened_checks;
	lists[MI] = &multis;
	lists[PI] = &parti;
	lists[CI] = &next_checks;
	lists[OI] = &opened_checks;
	int my_index[NITERS];
	int index_counts[NITERS];
	const float& myradius1 = myradius; // + h;
	const float myradius2 = SINK_BIAS * myradius1;
	int ninteractions = iamleaf ? 3 : 2;
	for (int type = 0; type < ninteractions; type++) {
		const bool ewald_dist = type == PC_PP_EWALD || type == CC_CP_EWALD;
		auto& checks = ewald_dist ? params.echecks : shmem.dchecks;
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
		const float& th = constant.th;
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
						const auto other_pos = check.get_pos();
						const float other_radius = check.get_radius();
						for (int dim = 0; dim < NDIM; dim++) {                         // 3
							dx[dim] = distance(other_pos[dim], mypos[dim]);
						}
						float d2 = fmaf(dx[0], dx[0], fmaf(dx[1], dx[1], sqr(dx[2]))); // 5
						if (ewald_dist) {
							d2 = fmaxf(d2, EWALD_MIN_DIST2); // 5
						}
						const float R1 = sqr(other_radius + myradius2 + th);                 // 2
						const float R2 = sqr(other_radius * theta + myradius2 + th); // 3
						const float R3 = sqr(other_radius + myradius1 * theta + th); // 3
						const float theta2d2 = theta2 * d2; // 1
						const int far1 = R1 < theta2d2;                 // 1
						const int far2 = R2 < theta2d2;                 // 1
						const int far3 = R3 < theta2d2;                 // 1
						const int isleaf = check.is_leaf();
						interacts++;
						flops += 27;
						const bool mi = far1 || (direct && far3/* && other_nparts >= MIN_PC_PARTS*/);
						const bool pi = (far2 || direct) && isleaf;
						list_index = (1 - mi) * (pi * PI + (1 - pi) * (isleaf * OI + (1 - isleaf) * CI));
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
		auto &tmp_parti = shmem.opened_checks;
		if (type == PC_PP_DIRECT) {
			auto &sinks = shmem.sink;
			const int pmax = max(((parti.size() - 1) / KICK_BLOCK_SIZE + 1) * KICK_BLOCK_SIZE, 0);
			for (int k = tid; k < myparts.second - myparts.first; k += KICK_BLOCK_SIZE) {
				const auto index = k + myparts.first;
				if (parts.rung(index) >= constant.rung || constant.full_eval) {
					for (int dim = 0; dim < NDIM; dim++) {
						sinks[dim][k] = parts.pos(dim, index);
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
					const auto other_pos = parti[j].get_pos();
					const auto other_radius = parti[j].get_radius();
					const auto parti_parts = parti[j].get_parts();
					const auto other_nparts = parti_parts.second - parti_parts.first;
					bool res = false;
					const int sz = myparts.second - myparts.first;
					if (other_nparts < MIN_PC_PARTS) {
						res = true;
					} else {
						for (int k = 0; k < sz; k++) {
							const auto this_rung = parts.rung(k + myparts.first);
							if (this_rung >= constant.rung || constant.full_eval) {
								float dx0 = distance(other_pos[0], sinks[0][k]);
								float dy0 = distance(other_pos[1], sinks[1][k]);
								float dz0 = distance(other_pos[2], sinks[2][k]);
								float d2 = fma(dx0, dx0, fma(dy0, dy0, sqr(dz0)));
								res = sqr(other_radius + th) > d2 * theta2;
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
	if (!shmem.self.is_leaf()) {
		const auto children = tptr.get_children();
		const auto left_active = children[LEFT].get_active_parts();
		const auto right_active = children[RIGHT].get_active_parts();
		if (tid == 0) {
			shmem.depth++;
			params.L[shmem.depth] = L;
			params.Lpos[shmem.depth] = mypos;
		}
		if ((left_active && right_active) || constant.full_eval) {
			shmem.dchecks.push_top();
			params.echecks.push_top();
			if (tid == 0) {
				shmem.self = children[LEFT];
			}
			__syncwarp();
			cuda_kick(params_ptr);
			if (tid == 0) {
				params.L[shmem.depth] = L;
				shmem.self = children[RIGHT];
			}
			shmem.dchecks.pop_top();
			params.echecks.pop_top();
			__syncwarp();
			cuda_kick(params_ptr);
		} else if (left_active) {
			if (tid == 0) {
				shmem.self = children[LEFT];
			}
			__syncwarp();
			cuda_kick(params_ptr);
		} else if (right_active) {
			if (tid == 0) {
				shmem.self = children[RIGHT];
			}
			__syncwarp();
			cuda_kick(params_ptr);
		}
		if (tid == 0) {
			shmem.depth--;
		}
		//   printf( "%li\n", rc.flops);
	} else {
		const float& invlog2 = constant.invlog2;
		const float& GM = constant.GM;
		const float& tfactor = constant.tfactor;
		const float& logt0 = constant.logt0;
		const float& halft0 = constant.halft0;
		const auto& minrung = constant.minrung;
		for (int k = tid; k < myparts.second - myparts.first; k += KICK_BLOCK_SIZE) {
			const auto this_rung = parts.rung(k + myparts.first);
			if (this_rung >= constant.rung || constant.full_eval) {
				array<float, NDIM> g;
				float this_phi;
				for (int dim = 0; dim < NDIM; dim++) {
					const auto x2 = mypos[dim];
					const auto x1 = parts.pos(dim, k + myparts.first);
					dx[dim] = distance(x1, x2);
				}
				shift_expansion(L, g, this_phi, dx, constant.full_eval);
				for (int dim = 0; dim < NDIM; dim++) {
					F[dim][k] += g[dim];
				}
				phi[k] += this_phi;
				for (int dim = 0; dim < NDIM; dim++) {
					F[dim][k] *= GM;
				}
				phi[k] *= GM;
#ifdef TEST_FORCE
				for (int dim = 0; dim < NDIM; dim++) {
					parts.force(dim, k + myparts.first) = F[dim][k];
				}
				parts.pot(k + myparts.first) = phi[k];
#endif
				if (this_rung >= constant.rung) {
					float dt = halft0 * rung_dt[this_rung];
					const auto index = k + myparts.first;
					auto& vx = parts.vel(0,index);
					auto& vy = parts.vel(1,index);
					auto& vz = parts.vel(2,index);
					if (!constant.first) {
						vx = fmaf(dt, F[0][k], vx);
						vy = fmaf(dt, F[1][k], vy);
						vz = fmaf(dt, F[2][k], vz);
					}
					const float fmag2 = fmaf(F[0][k], F[0][k], fmaf(F[1][k], F[1][k], sqr(F[2][k])));
					if (fmag2 != 0.f) {
						dt = fminf(tfactor * rsqrt(sqrtf(fmag2)), params.t0);
					} else {
						dt = 1.0e+30;
					}
					const int new_rung = fmaxf(fmaxf(ceilf((logt0 - logf(dt)) * invlog2), this_rung - 1), minrung);
					dt = halft0 * rung_dt[new_rung];
					vx = fmaf(dt, F[0][k], vx);
					vy = fmaf(dt, F[1][k], vy);
					vz = fmaf(dt, F[2][k], vz);
					parts.set_rung(new_rung, index);
				}
				if (constant.full_eval) {
					kick_return_update_pot_gpu(phi[k], F[0][k], F[1][k], F[2][k]);
				}
			}
			kick_return_update_rung_gpu(parts.rung(k + myparts.first));
		}
	}
	if (constant.full_eval) {
		kick_return_update_interactions_gpu(KR_OP, interacts, flops);
	}
}

CUDA_KERNEL cuda_set_kick_params_kernel(particle_set *p) {
	if (threadIdx.x == 0) {
		parts = p;
	}
}

void tree::show_timings() {
}

void tree::cuda_set_kick_params(particle_set *p) {
	cuda_set_kick_params_kernel<<<1,1>>>(p);
	CUDA_CHECK(cudaDeviceSynchronize());
}

CUDA_KERNEL cuda_kick_kernel(kick_params_type *params) {
	const int &bid = blockIdx.x;
	__shared__
	extern int shmem_ptr[];
	cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
	if (threadIdx.x == 0) {
		memcpy(&shmem.part_interactions, &(params[bid].part_interactions), sizeof(vector<tree_ptr> ));
		memcpy(&shmem.multi_interactions, &(params[bid].multi_interactions), sizeof(vector<tree_ptr> ));
		memcpy(&shmem.next_checks, &(params[bid].next_checks), sizeof(vector<tree_ptr> ));
		memcpy(&shmem.opened_checks, &(params[bid].opened_checks), sizeof(vector<tree_ptr> ));
		memcpy(&shmem.dchecks, &(params[bid].dchecks), sizeof(stack_vector<tree_ptr> ));
		shmem.self = params[bid].tptr;
		shmem.depth = params[bid].depth;
	}
	__syncthreads();
	cuda_kick(params + bid);
	__syncthreads();
	if (threadIdx.x == 0) {
		atomicMax(&list_sizes.multi, shmem.multi_interactions.capacity());
		atomicMax(&list_sizes.part, shmem.part_interactions.capacity());
		atomicMax(&list_sizes.next, shmem.next_checks.capacity());
		atomicMax(&list_sizes.open, shmem.opened_checks.capacity());
//		params[bid].kick_params_type::~kick_params_type();
		shmem.part_interactions.~vector<tree_ptr>();
		shmem.multi_interactions.~vector<tree_ptr>();
		shmem.next_checks.~vector<tree_ptr>();
		shmem.opened_checks.~vector<tree_ptr>();
		shmem.dchecks.~stack_vector<tree_ptr>();
		params[bid].echecks.~stack_vector<tree_ptr>();
	}

}

list_sizes_t get_list_sizes() {
	return list_sizes;
}

void reset_list_sizes() {
	list_sizes.multi = list_sizes.part = list_sizes.open = list_sizes.next = 0;
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

