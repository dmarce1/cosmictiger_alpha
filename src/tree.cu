struct ewald_indices;
struct periodic_parts;
#include <cosmictiger/cuda.hpp>
#include <stack>

__device__ ewald_indices *four_indices_ptr;
__device__ ewald_indices *real_indices_ptr;
__device__ periodic_parts *periodic_parts_ptr;

#define TREECU
#include <cosmictiger/tree.hpp>
#include <cosmictiger/interactions.hpp>
#include <functional>
#include <cosmictiger/gravity.hpp>

//CUDA_KERNEL cuda_kick()

#define CC_CP_DIRECT 0
#define CC_CP_EWALD 1
#define PC_PP_DIRECT 2
#define PC_PP_EWALD 3
#define N_INTERACTION_TYPES 4

CUDA_DEVICE particle_set *parts;

__managed__ double pp_interaction_time = 0;
__managed__ double pc_interaction_time = 0;
__managed__ double cp_interaction_time = 0;
__managed__ double cc_interaction_time = 0;
__managed__ double ewald_interaction_time = 0;
__managed__ double total_time = 0;

#define MI 0
#define PI 1
#define CI 2
#define OI 3

void show_timings() {
	const auto walk_time = total_time - pp_interaction_time - pc_interaction_time - cp_interaction_time
			- cc_interaction_time - ewald_interaction_time;
	printf("%e %e %e %e %e %e\n", walk_time / total_time, pp_interaction_time / total_time,
			pc_interaction_time / total_time, cp_interaction_time / total_time, cc_interaction_time / total_time,
			ewald_interaction_time / total_time);
	total_time = ewald_interaction_time = pp_interaction_time = pc_interaction_time = cp_interaction_time =
			cc_interaction_time = 0.0;

}

CUDA_DEVICE kick_return cuda_kick(kick_params_type * params_ptr) {
	kick_params_type &params = *params_ptr;
	__shared__ volatile
	extern int shmem_ptr[];
	cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
	//  printf( "%i\n", params_ptr->depth);
	//   if( params_ptr->depth > TREE_MAX_DEPTH || params_ptr->depth < 0 ) {
//      printf( "%li\n", params_ptr->depth);
	//  }
	tree_ptr tptr = params.tptr;
	tree& me = *((tree*) params.tptr);
	const int &tid = threadIdx.x;
	kick_return rc;
	auto &F = params.F;
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
		shift_expansion(L, dx);
	}

#ifdef COUNT_FLOPS
	int flops = 0;
#endif
	if (((tree*) tptr)->children[0].ptr == 0) {
		for (int k = tid; k < MAX_BUCKET_SIZE; k += KICK_BLOCK_SIZE) {
			for (int dim = 0; dim < NDIM; dim++) {
				F[dim][k] = 0.f;
			}
		}
		__syncwarp();
	}
	const auto& myparts = ((tree*) params.tptr)->parts;
	auto &indices = shmem.indices;
	const auto& h = params.hsoft;
	auto &count = shmem.count;

	const auto theta = params.theta;
	const auto theta2 = params.theta * params.theta;
	array<vector<tree_ptr>*, NITERS> lists;
	auto &multis = params.multi_interactions;
	auto &parti = params.part_interactions;
	auto &next_checks = params.next_checks;
	auto &opened_checks = params.opened_checks;
	lists[MI] = &multis;
	lists[PI] = &parti;
	lists[CI] = &next_checks;
	lists[OI] = &opened_checks;
	const auto myradius1 = me.radius + h;
	const auto myradius2 = SINK_BIAS * myradius1;
	const auto &mypos = me.pos;
	const bool iamleaf = me.children[0].ptr == 0;
	int ninteractions = iamleaf ? 4 : 2;
	const auto tp1 = tid + 1;
	for (int type = 0; type < ninteractions; type++) {
		const bool ewald_dist = type == PC_PP_EWALD || type == CC_CP_EWALD;
		auto& checks = ewald_dist ? params.echecks : params.dchecks;
		const bool direct = type == PC_PP_EWALD || type == PC_PP_DIRECT;
		if (tid < NITERS) {
			count[tid] = 0;
		}
		for (int i = 0; i < NITERS; i++) {
			lists[i]->resize(0);
		}
		__syncwarp();
		int check_count;
		do {
			check_count = checks.size();
			flops += check_count * FLOPS_OPEN;
			if (check_count) {
				const int cimax = ((check_count - 1) / KICK_BLOCK_SIZE + 1) * KICK_BLOCK_SIZE;
				for (int ci = tid; ci < cimax; ci += KICK_BLOCK_SIZE) {
					for (int i = 0; i < NITERS; i++) {
						indices[i][tp1] = 0;
					}
					__syncwarp();
					if (tid < NITERS) {
						indices[tid][0] = 0;
					}
					__syncwarp();
					const auto h = params.hsoft;
					if (ci < check_count) {
						auto &check = checks[ci];
						const auto &other_radius = ((const tree*) check)->radius;
						const auto &other_pos = ((const tree*) check)->pos;
						array<float, NDIM> dist;
						for (int dim = 0; dim < NDIM; dim++) {                         // 3
							dist[dim] = distance(other_pos[dim], mypos[dim]);
						}
						float d2 = fmaf(dist[0], dist[0], fmaf(dist[1], dist[1], sqr(dist[2])));
						d2 = (1 - int(ewald_dist)) * d2 + int(ewald_dist) * fmaxf(d2, EWALD_MIN_DIST2);
						const auto other_radius_ph = other_radius + h;
						const auto R1 = sqr(other_radius_ph + myradius2);                 // 2
						const auto R2 = sqr(other_radius_ph * theta + myradius2);
						const auto R3 = sqr(other_radius_ph + myradius1 * theta);
						const auto theta2d2 = theta2 * d2;
						const bool far1 = R1 < theta2d2;                 // 2
						const bool far2 = R2 < theta2d2;
						const bool far3 = R3 < theta2d2;
						const bool isleaf = ((const tree*) check)->children[0].ptr == 0;
						const bool mi = far1 || (direct && far3);
						const bool pi = (far2 || direct) && isleaf;
						list_index = int(mi) * MI
								+ (1 - int(mi)) * (int(pi) * PI + (1 - int(pi)) * (int(isleaf) * OI + (1 - int(isleaf)) * CI));
						indices[list_index][tp1] = 1;
					}
					__syncwarp();
					for (int P = 1; P < KICK_BLOCK_SIZE; P *= 2) {
						array<int, NITERS> tmp;
						if (tid - P + 1 >= 0) {
							for (int i = 0; i < NITERS; i++) {
								tmp[i] = indices[i][tid - P + 1];
							}
						}
						__syncwarp();
						if (tid - P + 1 >= 0) {
							for (int i = 0; i < NITERS; i++) {
								indices[i][tp1] += tmp[i];
							}
						}
						__syncwarp();
					}
					__syncwarp();
					for (int i = 0; i < NITERS; i++) {
						assert(indices[i][tid] <= indices[i][tp1]);
						lists[i]->resize(count[i] + indices[i][KICK_BLOCK_SIZE]);
					}
					if (ci < check_count) {
						const auto &check = checks[ci];
						assert(count[list_index] + indices[list_index][tid] >= 0);
						//          printf( "%i %i\n",(*lists[list_index]).size(), count[list_index] + indices[list_index][tid] );
						(*lists[list_index])[count[list_index] + indices[list_index][tid]] = check;
					}
					__syncwarp();
					if (tid < NITERS) {
						count[tid] += indices[tid][KICK_BLOCK_SIZE];
					}
					__syncwarp();
				}
				__syncwarp();
				auto& countCI = count[CI];
				auto& countOI = count[OI];
				check_count = 2 * countCI + countOI;
				checks.resize(check_count);
				for (int i = tid; i < countCI; i += KICK_BLOCK_SIZE) {
					const auto base = 2 * i;
					const auto children = next_checks[i].get_children();
					for (int j = 0; j < NCHILD; j++) {
						checks[base + j] = children[j];
					}
				}
				__syncwarp();
				const auto base = 2 * countCI;
				for (int i = tid; i < countOI; i += KICK_BLOCK_SIZE) {
					checks[base + i] = opened_checks[i];
				}
				__syncwarp();
				if (tid == 0) {
					countCI = 0;
					countOI = 0;
				}
				__syncwarp();
				next_checks.resize(0);
				opened_checks.resize(0);
			}
		} while (direct && check_count);
		__syncwarp();
		auto &tmp_parti = params.tmp;
		auto tm = clock64();
		if (type == PC_PP_DIRECT) {
			auto &rungs = shmem.rungs;
			auto &sinks = shmem.sink;
			const int pmax = max(((parti.size() - 1) / KICK_BLOCK_SIZE + 1) * KICK_BLOCK_SIZE, 0);
			for (int k = tid; k < myparts.second - myparts.first; k += KICK_BLOCK_SIZE) {
				rungs[k] = parts->rung(k + myparts.first);
				if (rungs[k] >= params.rung) {
					for (int dim = 0; dim < NDIM; dim++) {
						sinks[dim][k] = parts->pos(dim, myparts.first + k);
					}
				}
			}
			tmp_parti.resize(0);
			for (int j = tid; j < pmax; j += KICK_BLOCK_SIZE) {
				if (tid == 0) {
					indices[PI][0] = indices[MI][0] = 0;
				}
				indices[PI][tp1] = 0;
				indices[MI][tp1] = 0;
				const auto& other = *((tree*) parti[j]);
				const size_t& first = myparts.first;
				const size_t& last = myparts.second;
				list_index = -1;
				const auto hfac = (1.f + SINK_BIAS) * params.hsoft;
				if (j < parti.size()) {
					bool res = false;
					for (int k = 0; k < last - first; k++) {
						const auto this_rung = rungs[k];
						if (this_rung >= params.rung) {
							float dx0 = distance(other.pos[0], sinks[0][k]);
							float dy0 = distance(other.pos[1], sinks[1][k]);
							float dz0 = distance(other.pos[2], sinks[2][k]);
							float d2 = fma(dx0, dx0, fma(dy0, dy0, sqr(dz0)));
							res = sqr(other.radius + hfac) > d2 * theta2;
							if (res) {
								break;
							}
						}
					}
					list_index = res ? PI : MI;
					indices[list_index][tp1] = 1;
				}
				for (int P = 1; P < KICK_BLOCK_SIZE; P *= 2) {
					array<int, 2> tmp;
					__syncwarp();
					if (tid - P + 1 >= 0) {
						tmp[0] = indices[0][tid - P + 1];
						tmp[1] = indices[1][tid - P + 1];
					}
					__syncwarp();
					if (tid - P + 1 >= 0) {
						indices[0][tp1] += tmp[0];
						indices[1][tp1] += tmp[1];
					}
				}
				const auto part_cnt = tmp_parti.size();
				const auto mult_cnt = multis.size();
				__syncwarp();
				tmp_parti.resize(part_cnt + indices[PI][KICK_BLOCK_SIZE]);
				multis.resize(mult_cnt + indices[MI][KICK_BLOCK_SIZE]);
				if (list_index == PI) {
					tmp_parti[part_cnt + indices[PI][tid]] = parti[j];
				} else if (list_index == MI) {
					multis[mult_cnt + indices[MI][tid]] = parti[j];
				}
				__syncwarp();
			}
		}
		switch (type) {
		case PC_PP_DIRECT:
			tm = clock64();
			flops += cuda_pc_interactions(parts, multis, params_ptr);
			if (tid == 0) {
				atomicAdd(&pc_interaction_time, (double) (clock64() - tm));
			}
			tm = clock64();
			flops += cuda_pp_interactions(parts, tmp_parti, params_ptr);
			if (tid == 0) {
				atomicAdd(&pp_interaction_time, (double) (clock64() - tm));
			}
			break;
		case CC_CP_DIRECT:
			tm = clock64();
			flops += cuda_cc_interactions(parts, multis, params_ptr);
			if (tid == 0) {
				atomicAdd(&cc_interaction_time, (double) (clock64() - tm));
			}
			tm = clock64();
			flops += cuda_cp_interactions(parts, parti, params_ptr);
			if (tid == 0) {
				atomicAdd(&cp_interaction_time, (double) (clock64() - tm));
			}
			break;

		case PC_PP_EWALD:
			if (count[PI] > 0) {
				//        printf( "PP Ewald should not exist\n");
				//  __trap();
			}
			if (count[MI] > 0) {
				//      printf( "PC Ewald should not exist\n");
				//   __trap();
			}
			break;
		case CC_CP_EWALD:
#ifndef PERIODIC_OFF
			if (count[PI] > 0) {
				printf("CP Ewald should not exist\n");
				//     __trap();
			}
			if (count[MI] > 0) {
				tm = clock64();
				flops += cuda_ewald_cc_interactions(parts, params_ptr, &shmem.Lreduce);
				if (tid == 0) {
					atomicAdd(&ewald_interaction_time, (double) (clock64() - tm));
				}
				tm = clock64();
			}
#endif
			break;
		}
	}
	rc.flops = flops;
	if (!(((tree*) tptr)->children[0].ptr == 0)) {
		params.dchecks.push_top();
		params.echecks.push_top();
		if (tid == 0) {
			params.depth++;
			params.L[params.depth] = L;
			params.Lpos[params.depth] = me.pos;
			params.tptr = ((tree*) tptr)->children[RIGHT];
		}
		__syncwarp();
		kick_return rc1 = cuda_kick(params_ptr);
		if (tid == 0) {
			params.L[params.depth] = L;
			params.tptr = ((tree*) tptr)->children[LEFT];
		}
		__syncwarp();
		params.dchecks.pop_top();
		params.echecks.pop_top();
		kick_return rc2 = cuda_kick(params_ptr);
		if (tid == 0) {
			params.depth--;
		}
		__syncwarp();
		rc.rung = max(rc1.rung, rc2.rung);
		rc.flops += rc1.flops + rc2.flops;
		//   printf( "%li\n", rc.flops);
	} else {
		auto& rungs = shmem.rungs;
		rungs[tid] = 0;
		const auto invlog2 = 1.0f / logf(2);
		for (int k = tid; k < myparts.second - myparts.first; k += KICK_BLOCK_SIZE) {
			const auto this_rung = parts->rung(k + myparts.first);
			if (this_rung >= params.rung) {

				array<float, NDIM> g;
				float phi;
				array<float, NDIM> dx;
				for (int dim = 0; dim < NDIM; dim++) {
					const auto x2 = me.pos[dim];
					const auto x1 = parts->pos(dim, k + myparts.first);
					dx[dim] = distance(x1, x2);
				}
				shift_expansion(L, g, phi, dx);
				for (int dim = 0; dim < NDIM; dim++) {
					F[dim][k] += g[dim];
				}
#ifdef TEST_FORCE
				for (int dim = 0; dim < NDIM; dim++) {
					parts->force(dim, k + myparts.first) = F[dim][k];
				}
#endif
				float dt = params.t0 / (1 << this_rung);
				if (!params.first) {
					for (int dim = 0; dim < NDIM; dim++) {
						parts->vel(dim, k + myparts.first) += 0.5 * dt * F[dim][k];
					}
				}
				float fmag = 0.0;
				for (int dim = 0; dim < NDIM; dim++) {
					fmag += sqr(F[dim][k]);
				}
				fmag = sqrtf(fmag);
				//   printf( "%e\n", fmag);
				assert(fmag > 0.0);
				dt = fminf(sqrt(params.scale * params.eta / fmag), params.t0);
				int new_rung = fmaxf(fmaxf(ceil(logf(params.t0 / dt) * invlog2), this_rung - 1), params.rung);
				dt = params.t0 / (1 << new_rung);
				for (int dim = 0; dim < NDIM; dim++) {
					parts->vel(dim, k + myparts.first) += 0.5 * dt * F[dim][k];
				}
				rungs[tid] = fmaxf(rungs[tid], new_rung);
				parts->set_rung(new_rung, k + myparts.first);
			}
		}
		__syncwarp();
		for (int P = KICK_BLOCK_SIZE / 2; P >= 1; P /= 2) {
			if (tid < P) {
				rungs[tid] = fmaxf(rungs[tid], rungs[tid + P]);
			}
			__syncwarp();
		}
		rc.rung = rungs[0];
	}
	return rc;
}

CUDA_KERNEL cuda_set_kick_params_kernel(particle_set *p, ewald_indices *real_indices, ewald_indices *four_indices,
		periodic_parts *periodic_parts) {
	if (threadIdx.x == 0) {
		parts = p;
		four_indices_ptr = four_indices;
		real_indices_ptr = real_indices;
		periodic_parts_ptr = periodic_parts;
		expansion_init();

	}
}
void tree::cuda_set_kick_params(particle_set *p, ewald_indices *real_indices, ewald_indices *four_indices,
		periodic_parts *parts) {
	cuda_set_kick_params_kernel<<<1,1>>>(p,real_indices, four_indices, parts);
	CUDA_CHECK(cudaDeviceSynchronize());
	expansion_init_cpu();
	if( tree::real_indices_ptr == nullptr ) {
		tree::real_indices_ptr = new ewald_indices(EWALD_NREAL, false);
		tree::four_indices_ptr = new ewald_indices(EWALD_NFOUR, true);
		tree::periodic_parts_ptr = new periodic_parts();
	}
}

#ifdef TIMINGS
extern __managed__ double pp_crit1_time;
extern __managed__ double pp_crit2_time;
#endif

CUDA_KERNEL cuda_kick_kernel(kick_return *res, kick_params_type *params) {
	const int &bid = blockIdx.x;
#ifdef TIMINGS
	auto tm = clock64();
#endif
	auto tm = clock64();
	res[bid] = cuda_kick(params + bid);
	__syncwarp();
	if (threadIdx.x == 0) {
		//     printf( "Kick done\n");
		params[bid].kick_params_type::~kick_params_type();
		atomicAdd(&total_time, (double) clock64() - tm);
	}

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

CUDA_KERNEL cuda_ewald_cc_kernel(kick_params_type **params_ptr) {
	__shared__
	volatile
	extern int shmem_ptr[];
	cuda_ewald_shmem &shmem = *((cuda_ewald_shmem*) (shmem_ptr));
	const int &bid = blockIdx.x;
	auto pptr = params_ptr[bid];
	auto rc = cuda_ewald_cc_interactions(parts, pptr, &shmem.Lreduce);
	__syncwarp();
	if (threadIdx.x == 0) {
		params_ptr[bid]->flops = rc;
	}
}

std::function<bool()> cuda_execute_ewald_kernel(kick_params_type **params_ptr, int grid_size) {
	auto stream = get_stream();
	/***/cuda_ewald_cc_kernel<<<grid_size,KICK_BLOCK_SIZE,sizeof(cuda_ewald_shmem),stream>>>(params_ptr);

	struct cuda_ewald_future_shared {
		cudaStream_t stream;
		int grid_size;
		mutable bool ready;
	public:
		cuda_ewald_future_shared() {
			ready = false;
		}
		bool operator()() const {
			if (!ready) {
				if (cudaStreamQuery(stream) == cudaSuccess) {
					ready = true;
					CUDA_CHECK(cudaStreamSynchronize(stream));
					cleanup_stream(stream);
				}
			}
			return ready;
		}
	};

	cuda_ewald_future_shared fut;
	fut.stream = stream;
	fut.grid_size = grid_size;
	std::function<bool()> ready_func = [fut]() {
		return fut();
	};
	return ready_func;
}

kick_return* cuda_execute_kick_kernel(kick_params_type *params, int grid_size, cudaStream_t stream) {
	const size_t shmemsize = sizeof(cuda_kick_shmem);
	unified_allocator alloc;
	kick_return *returns = (kick_return*) alloc.allocate(grid_size * sizeof(kick_return));
	/***************************************************************************************************************************************************/
	/**/cuda_kick_kernel<<<grid_size, KICK_BLOCK_SIZE, shmemsize, stream>>>(returns,params);/**/
	/***************************************************************************************************************************************************/
	return returns;
}

