/*
 * gravity.cu
 *
 *  Created on: Feb 10, 2021
 *      Author: dmarce1
 */

#include <cosmictiger/gravity.hpp>
#include <cosmictiger/array.hpp>
#include <cosmictiger/kick_return.hpp>
#include <cosmictiger/cuda.hpp>

CUDA_DEVICE void cuda_cc_interactions(kick_params_type *params_ptr, eval_type etype) {
	kick_params_type &params = *params_ptr;
	const auto& multis = params.multi_interactions;
	const int &tid = threadIdx.x;
	__shared__
	extern int shmem_ptr[];
	cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
//	auto &Lreduce = shmem.Lreduce;
	if (multis.size() == 0) {
		return;
	}
	expansion<float> L = shmem.expanse2[tid];
	expansion<float> D = shmem.expanse1[tid];
	int flops = 0;
	int interacts = 0;
	for (int i = 0; i < LP; i++) {
		L[i] = 0.0f;
	}
	const auto &pos = ((tree*) params.tptr)->pos;
	const int sz = multis.size();
	for (int i = tid; i < sz; i += KICK_BLOCK_SIZE) {
		const multipole mpole = ((tree*) multis[i])->multi;
		array<float, NDIM> fpos;
		for (int dim = 0; dim < NDIM; dim++) {
			fpos[dim] = distance(pos[dim], ((tree*) multis[i])->pos[dim]);
		}
		flops += 6;
		if (etype == DIRECT) {
			flops += green_direct(D, fpos);
		} else {
			flops += green_ewald(D, fpos);
		}
		flops += multipole_interaction(L, mpole, D);
		interacts++;
	}

	for (int P = KICK_BLOCK_SIZE / 2; P >= 1; P /= 2) {
		for (int i = 0; i < LP; i++) {
			L[i] += __shfl_xor_sync(0xffffffff, L[i], P);
		}
	}
	for (int i = tid; i < LP; i += KICK_BLOCK_SIZE) {
		params.L[params.depth][i] += L[i];
	}
	__syncwarp();
	if( params.full_eval) {
		kick_return_update_interactions_gpu(etype == DIRECT ? KR_CC : KR_EWCC, interacts, flops);
	}
}

CUDA_DEVICE void cuda_cp_interactions(kick_params_type *params_ptr) {
	kick_params_type &params = *params_ptr;
	particle_set *parts = params.particles;
	const auto& parti = params.part_interactions;
	const int &tid = threadIdx.x;
	__shared__
	extern int shmem_ptr[];
	cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
	//auto &Lreduce = shmem.Lreduce;
	if (parti.size() == 0) {
		return;
	}
	auto &sources = shmem.src;
	const auto &myparts = ((tree*) params.tptr)->parts;
	int part_index;
	int flops = 0;
	int interacts = 0;
	expansion<float> L;
	if (parti.size() > 0) {
		for (int j = 0; j < LP; j++) {
			L[j] = 0.0;
		}
		auto these_parts = ((tree*) parti[0])->parts;
		int i = 0;
		const auto &pos = ((tree*) params.tptr)->pos;
		const auto partsz = parti.size();
		while (i < partsz) {
			part_index = 0;
			while (part_index < KICK_PP_MAX && i < partsz) {
				while (i + 1 < partsz) {
					const auto other_tree = ((tree*) parti[i + 1]);
					if (these_parts.second == other_tree->parts.first) {
						these_parts.second = other_tree->parts.second;
						i++;
					} else {
						break;
					}
				}
				const size_t imin = these_parts.first;
				const size_t imax = min(these_parts.first + (KICK_PP_MAX - part_index), these_parts.second);
				const int sz = imax - imin;
				for (int j = tid; j < sz; j += KICK_BLOCK_SIZE) {
					for (int dim = 0; dim < NDIM; dim++) {
						sources[dim][part_index + j] = parts->pos(dim, j + imin);
					}
				}
				these_parts.first += sz;
				part_index += sz;
				if (these_parts.first == these_parts.second) {
					i++;
					if (i < parti.size()) {
						these_parts = ((tree*) parti[i])->parts;
					}
				}
			}
			for (int j = tid; j < part_index; j += KICK_BLOCK_SIZE) {
				array<float, NDIM> dx;
				for (int dim = 0; dim < NDIM; dim++) {
					dx[dim] = distance(pos[dim], sources[dim][j]);
				}
				expansion<float> D;
				flops += 3;
				flops += green_direct(D, dx);
				flops += multipole_interaction(L, D);
				interacts++;
			}
		}
		for (int P = KICK_BLOCK_SIZE / 2; P >= 1; P /= 2) {
			for (int i = 0; i < LP; i++) {
				L[i] += __shfl_xor_sync(0xffffffff, L[i], P);
			}
		}
		for (int i = tid; i < LP; i += KICK_BLOCK_SIZE) {
			params.L[params.depth][i] += L[i];
		}
	}
	__syncwarp();
	if( params.full_eval) {
		kick_return_update_interactions_gpu(KR_CP, interacts, flops);
	}
}

CUDA_DEVICE void cuda_pp_interactions(kick_params_type *params_ptr) {
	kick_params_type &params = *params_ptr;
	particle_set *parts = params.particles;
	const auto& parti = params.part_interactions;
	const int &tid = threadIdx.x;
	__shared__
	extern int shmem_ptr[];
	cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
//	auto &f = shmem.f;
	auto &F = params.F;
	auto &Phi = params.Phi;
	auto &rungs = shmem.rungs;
	auto &sources = shmem.src;
	auto &sinks = shmem.sink;
	const auto h = params.hsoft;
	const auto h2 = h * h;
	const auto hinv = 1.0f / h;
	int flops = 0;
	int interacts = 0;
	int part_index;
	if (parti.size() == 0) {
		return;
	}
//   printf( "%i\n", parti.size());
	const auto &myparts = ((tree*) params.tptr)->parts;
	const int nsinks = myparts.second - myparts.first;
	for (int i = tid; i < nsinks; i += KICK_BLOCK_SIZE) {
		rungs[i] = parts->rung(i + myparts.first);
		if (rungs[i] >= params.rung || params.full_eval) {
			for (int dim = 0; dim < NDIM; dim++) {
				sinks[dim][i] = parts->pos(dim, i + myparts.first);
			}
		}
	}
	int i = 0;
	auto these_parts = ((tree*) parti[0])->parts;
	const auto partsz = parti.size();
	float fx;
	float fy;
	float fz;
	float phi;
	float dx0, dx1, dx2;
	while (i < partsz) {
		part_index = 0;
		while (part_index < KICK_PP_MAX && i < partsz) {
			while (i + 1 < partsz) {
				const auto other_tree = ((tree*) parti[i + 1]);
				if (these_parts.second == other_tree->parts.first) {
					these_parts.second = other_tree->parts.second;
					i++;
				} else {
					break;
				}
			}
			const size_t imin = these_parts.first;
			const size_t imax = min(these_parts.first + (KICK_PP_MAX - part_index), these_parts.second);
			const int sz = imax - imin;
			for (int j = tid; j < sz; j += KICK_BLOCK_SIZE) {
				for (int dim = 0; dim < NDIM; dim++) {
					sources[dim][part_index + j] = parts->pos(dim, j + imin);
				}
			}
			these_parts.first += sz;
			part_index += sz;
			if (these_parts.first == these_parts.second) {
				i++;
				if (i < partsz) {
					these_parts = ((tree*) parti[i])->parts;
				}
			}
		}
		array<float, NDIM> dx;
		float r3inv, r1inv;
		__syncwarp();
		for (int k = 0; k < nsinks; k++) {
			if (rungs[k] >= params.rung || params.full_eval) {
				fx = 0.f;
				fy = 0.f;
				fz = 0.f;
				phi = 0.f;
				for (int j = tid; j < part_index; j += KICK_BLOCK_SIZE) {
					dx0 = distance(sinks[0][k], sources[0][j]);
					dx1 = distance(sinks[1][k], sources[1][j]);
					dx2 = distance(sinks[2][k], sources[2][j]);               // 3
					const auto r2 = fmaf(dx0, dx0, fmaf(dx1, dx1, sqr(dx2))); // 5
					if (r2 >= h2) {
						r1inv = rsqrt(r2);                                    // FLOP_RSQRT
						r3inv = r1inv * r1inv * r1inv;                        // 2
						flops += 2 + FLOP_RSQRT;
					} else {
						const float r1oh1 = sqrtf(r2) * hinv;              // 1 + FLOP_SQRT
						const float r2oh2 = r1oh1 * r1oh1;           // 1
						r3inv = +15.0f / 8.0f;
						r3inv = fmaf(r3inv, r2oh2, -21.0f / 4.0f);
						r3inv = fmaf(r3inv, r2oh2, +35.0f / 8.0f);
						r1inv = -5.0f / 16.0f;
						r1inv = fmaf(r1inv, r2oh2, 21.0f / 16.0f);
						r1inv = fmaf(r1inv, r2oh2, -35.0f / 16.0f);
						r1inv = fmaf(r1inv, r2oh2, 35.0f / 16.0f);
						flops += FLOP_SQRT + 16;
					}
					fx = fmaf(dx[0], r3inv, fx); // 2
					fy = fmaf(dx[1], r3inv, fy); // 2
					fz = fmaf(dx[2], r3inv, fz); // 2
					phi -= r1inv; // 1
					flops += 15;
					interacts++;
				}
				for (int P = KICK_BLOCK_SIZE / 2; P >= 1; P /= 2) {
					fx += __shfl_down_sync(0xffffffff, fx, P);
					fy += __shfl_down_sync(0xffffffff, fy, P);
					fz += __shfl_down_sync(0xffffffff, fz, P);
					phi += __shfl_down_sync(0xffffffff, phi, P);
				}
				if (tid == 0) {
					F[0][k] -= fx;
					F[1][k] -= fy;
					F[2][k] -= fz;
					Phi[k] += phi;
				}
			}
		}
	}
	__syncwarp();
	if( params.full_eval) {
		kick_return_update_interactions_gpu(KR_PP, interacts, flops);
	}
}

CUDA_DEVICE
void cuda_pc_interactions(kick_params_type *params_ptr) {
	kick_params_type &params = *params_ptr;
	particle_set *parts = params.particles;
	const auto& multis = params.multi_interactions;
	const int &tid = threadIdx.x;

	__shared__
	extern int shmem_ptr[];
	cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
	//auto &f = shmem.f;
	auto &F = params.F;
	auto &Phi = params.Phi;
	const auto &myparts = ((tree*) params.tptr)->parts;
	const int nparts = myparts.second - myparts.first;
	if (multis.size() == 0) {
		return;
	}
	auto &rungs = shmem.rungs;
	auto &sinks = shmem.sink;
	auto& msrcs = shmem.msrc;
	for (int i = tid; i < nparts; i += KICK_BLOCK_SIZE) {
		rungs[i] = parts->rung(i + myparts.first);
		if (rungs[i] >= params.rung || params.full_eval) {
			for (int dim = 0; dim < NDIM; dim++) {
				sinks[dim][i] = parts->pos(dim, myparts.first + i);
			}
		}
	}
	int interacts = 0;
	int flops = 0;
	array<float, NDIM> dx;
	array<float, NDIM + 1> Lforce;
	expansion<float> D;
	float fx;
	float fy;
	float fz;
	float phi;
	auto& dx0 = dx[0];
	auto& dx1 = dx[1];
	auto& dx2 = dx[2];
	int nsrc = 0;
	const int msize = sizeof(multipole_pos) / sizeof(float);
	const auto mmax = (((multis.size() - 1) / KICK_PC_MAX) + 1) * KICK_PC_MAX;
	for (int m = 0; m < mmax; m += KICK_PC_MAX) {
		nsrc = 0;
		for (int z = 0; z < KICK_PC_MAX; z++) {
			if (m + z < multis.size()) {
				const auto& other_ptr = ((tree*) multis[m + z]);
				const float* src = (float*) &other_ptr->multi;
				float* dst = (float*) &(msrcs[nsrc]);
				nsrc++;
				if (tid < msize) {
					dst[tid] = src[tid];
				}
			}
		}
		__syncwarp();
		for (int k = 0; k < nparts; k++) {
			if (rungs[k] >= params.rung || params.full_eval) {
				for (int i = 0; i < NDIM + 1; i++) {
					Lforce[i] = 0.f;
				}
				for (int i = tid; i < nsrc; i += KICK_BLOCK_SIZE) {
					const auto &source = msrcs[i].pos;
					dx0 = distance(sinks[0][k], source[0]);
					dx1 = distance(sinks[1][k], source[1]);
					dx2 = distance(sinks[2][k], source[2]);
					flops += 6;
					flops += green_direct(D, dx);
					flops += multipole_interaction(Lforce, msrcs[i].multi, D);
					interacts++;
				}
				phi = Lforce[0];
				fx = Lforce[1];
				fy = Lforce[2];
				fz = Lforce[3];
				for (int P = KICK_BLOCK_SIZE / 2; P >= 1; P /= 2) {
					fx += __shfl_down_sync(0xffffffff, fx, P);
					fy += __shfl_down_sync(0xffffffff, fy, P);
					fz += __shfl_down_sync(0xffffffff, fz, P);
					phi += __shfl_down_sync(0xffffffff, phi, P);
				}
				if (tid == 0) {
					F[0][k] -= fx;
					F[1][k] -= fy;
					F[2][k] -= fz;
					Phi[k] += phi;
				}
			}
		}
	}
	__syncwarp();
	if( params.full_eval) {
		kick_return_update_interactions_gpu(KR_PC, interacts, flops);
	}
}

#ifdef TEST_FORCE
CUDA_KERNEL cuda_pp_ewald_interactions(particle_set *parts, size_t *test_parts, float *ferr, float *fnorm, float* perr,
		float* pnorm, float GM);

#ifdef __CUDA_ARCH__
CUDA_KERNEL cuda_pp_ewald_interactions(particle_set *parts, size_t *test_parts, float *ferr, float *fnorm, float* perr, float* pnorm, float GM) {
	const int &tid = threadIdx.x;
	const int &bid = blockIdx.x;

	const auto index = test_parts[bid];
	array<fixed32, NDIM> sink;
	for (int dim = 0; dim < NDIM; dim++) {
		sink[dim] = parts->pos(dim, index);
	}
	const auto f_x = parts->force(0, index);
	const auto f_y = parts->force(1, index);
	const auto f_z = parts->force(2, index);
	__shared__ array<array<double, KICK_BLOCK_SIZE>, NDIM>
	f;
	__shared__ array<double,KICK_BLOCK_SIZE> phi;
	for (int dim = 0; dim < NDIM; dim++) {
		f[dim][tid] = 0.0;
	}
#ifdef PERIODIC_OFF
	phi[tid] = 0.0;
#else
	phi[tid] = -PHI0;
#endif
	for (size_t source = tid; source < parts->size(); source += KICK_BLOCK_SIZE) {
		if (source != index) {
			array<float, NDIM> X;
			for (int dim = 0; dim < NDIM; dim++) {
				const auto a = sink[dim];
				const auto b = parts->pos(dim, source);
				X[dim] = distance(a, b);
			}
#ifdef PERIODIC_OFF
			const auto r2 = fmaf(X[0], X[0], fmaf(X[1], X[1], sqr(X[2]))); // 5
			float r1inv, r3inv;
			r1inv = rsqrt(r2);// FLOP_RSQRT
			r3inv = r1inv * r1inv * r1inv;// 2
			f[0][tid] = fmaf(X[0], r3inv, f[0][tid]);// 2
			f[1][tid] = fmaf(X[1], r3inv, f[1][tid]);// 2
			f[2][tid] = fmaf(X[2], r3inv, f[2][tid]);// 2
			phi[tid] -= r1inv;// 1
#else
			const ewald_const econst;
			for (int i = 0; i < econst.nreal(); i++) {
				const auto n =econst.real_index(i);
				array<float, NDIM> dx;
				for (int dim = 0; dim < NDIM; dim++) {
					dx[dim] = X[dim] - n[dim];
				}
				const float r2 = sqr(dx[0]) + sqr(dx[1]) + sqr(dx[2]);
				if (r2 < (EWALD_REAL_CUTOFF2)) {  // 1
					const float r = sqrt(r2);// 1
					const float rinv = 1.f / r;// 2
					const float r2inv = rinv * rinv;// 1
					const float r3inv = r2inv * rinv;// 1
					const float exp0 = expf(-4.f * r2);// 26
					const float erfc0 = erfcf(2.f * r);// 10
					const float expfactor = 4.0 / sqrt(M_PI) * r * exp0;// 2
					const float d0 = -erfc0 * rinv;
					const float d1 = (expfactor + erfc0) * r3inv;// 2
					phi[tid] += d0;
					for (int dim = 0; dim < NDIM; dim++) {
						f[dim][tid] -= dx[dim] * d1;
					}
				}
			}
			for (int i = 0; i < econst.nfour(); i++) {
				const auto &h = econst.four_index(i);
				const auto &hpart = econst.four_expansion(i);
				const float hdotx = h[0] * X[0] + h[1] * X[1] + h[2] * X[2];
				float co = cosf(2.0 * M_PI * hdotx);
				float so = sinf(2.0 * M_PI * hdotx);
				phi[tid] += hpart() * co;
				for (int dim = 0; dim < NDIM; dim++) {
					f[dim][tid] -= hpart(dim) * so;
				}
			}
			phi[tid] += float(M_PI / 4.f);
#endif
		}
	}
	cuda_sync();
	for (int P = KICK_BLOCK_SIZE / 2; P >= 1; P /= 2) {
		if (tid < P) {
			for (int dim = 0; dim < NDIM; dim++) {
				f[dim][tid] += f[dim][tid + P];
			}
			phi[tid] += phi[tid + P];
		}
		cuda_sync();
	}
	for(int dim = 0; dim < NDIM; dim++) {
		f[dim][0] =GM * f[dim][0];
	}
	phi[0] = GM * phi[0];
	const auto f_ffm = sqrt(f_x * f_x + f_y * f_y + f_z * f_z);
	const auto f_dir = sqrt(sqr(f[0][0]) + sqr(f[1][0]) + sqr(f[2][0]));
	if (tid == 0) {
		fnorm[bid] = f_dir;
		pnorm[bid] = fabs(phi[0]);
		ferr[bid] = fabsf(f_ffm - f_dir);
		perr[bid] = fabs(parts->pot(index) - phi[0]);
//		printf( "%e %e\n",f[0][0], f_x);
	}
}
#endif

void cuda_compare_with_direct(particle_set *parts) {
	size_t *test_parts;
	float *ferrs;
	float *fnorms;
	float *perrs;
	float *pnorms;
	CUDA_MALLOC(test_parts, N_TEST_PARTS);
	CUDA_MALLOC(ferrs, N_TEST_PARTS);
	CUDA_MALLOC(fnorms, N_TEST_PARTS);
	CUDA_MALLOC(perrs, N_TEST_PARTS);
	CUDA_MALLOC(pnorms, N_TEST_PARTS);
	const size_t nparts = parts->size();
	for (int i = 0; i < N_TEST_PARTS; i++) {
		test_parts[i] = rand() % nparts;
	}
	cuda_pp_ewald_interactions<<<N_TEST_PARTS,KICK_BLOCK_SIZE>>>(parts, test_parts, ferrs, fnorms, perrs, pnorms, global().opts.G * global().opts.M);
	CUDA_CHECK(cudaDeviceSynchronize());
	float favg_err = 0.0;
	float fnorm = 0.0;
	float ferr_max = 0.0;
	float pavg_err = 0.0;
	float pnorm = 0.0;
	float perr_max = 0.0;
	for (int i = 0; i < N_TEST_PARTS; i++) {
		favg_err += ferrs[i];
		ferr_max = fmaxf(ferr_max, ferrs[i]);
		fnorm += fnorms[i];
	}
	for (int i = 0; i < N_TEST_PARTS; i++) {
		pavg_err += perrs[i];
		perr_max = fmaxf(perr_max, perrs[i]);
		pnorm += pnorms[i];
	}
//	avg_err /= N_TEST_PARTS;
	ferr_max /= (fnorm / N_TEST_PARTS);
	favg_err /= fnorm;
	perr_max /= (pnorm / N_TEST_PARTS);
	pavg_err /= pnorm;
	printf("Avg Force Error is %e\n", favg_err);
	printf("Max Froce Error is %e\n", ferr_max);
	printf("Avg Pot Error is %e\n", pavg_err);
	printf("Max Pot Error is %e\n", perr_max);
	CUDA_FREE(fnorms);
	CUDA_FREE(ferrs);
	CUDA_FREE(pnorms);
	CUDA_FREE(perrs);
	CUDA_FREE(test_parts);
}

#endif
