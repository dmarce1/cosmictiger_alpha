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

CUDA_DEVICE void cuda_cc_interactions(particle_set *parts, const vector<tree_ptr> &multis,
		kick_params_type *params_ptr) {

	kick_params_type &params = *params_ptr;
	const int &tid = threadIdx.x;
	volatile __shared__
	extern int shmem_ptr[];
	cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
	auto &Lreduce = shmem.Lreduce;
	if (multis.size() == 0) {
		return;
	}
	expansion<float> L;
	int flops = 0;
	int interacts = 0;
	for (int i = 0; i < LP; i++) {
		L[i] = 0.0f;
	}
	const auto &pos = ((tree*) params.tptr)->pos;
	const int sz = multis.size();
	expansion<float> D;
	for (int i = tid; i < sz; i += KICK_BLOCK_SIZE) {
		const multipole mpole = ((tree*) multis[i])->multi;
		array<float, NDIM> fpos;
		for (int dim = 0; dim < NDIM; dim++) {
			fpos[dim] = distance(pos[dim], ((tree*) multis[i])->pos[dim]);
		}
		flops += 6;
		flops += green_direct(D, fpos);
		flops += multipole_interaction(L, mpole, D);
		interacts++;
	}
	for (int i = 0; i < LP; i++) {
		Lreduce[tid] = L[i];
		cuda_sync();
		for (int P = KICK_BLOCK_SIZE / 2; P >= 1; P /= 2) {
			if (tid < P) {
				Lreduce[tid] += Lreduce[tid + P];
			}
			cuda_sync();
		}
		if (tid == 0) {
			params.L[params.depth][i] += Lreduce[0];
		}
	}
	kick_return_update_interactions_gpu(KR_CC, interacts, flops);
}

#ifdef __CUDA_ARCH__

CUDA_DEVICE void cuda_ewald_cc_interactions(particle_set *parts, kick_params_type *params_ptr,
		array<float, KICK_BLOCK_SIZE> *lptr) {

	kick_params_type &params = *params_ptr;
	const int &tid = threadIdx.x;
	auto &Lreduce = *lptr;
	auto &multis = params.multi_interactions;
	if( multis.size() == 0 ) {
		return;
	}
	expansion<float> L;
	for (int i = 0; i < LP; i++) {
		L[i] = 0.0;
	}
	int interacts = 0;
	int flops = 0;
	const auto &pos = ((tree*) params.tptr)->pos;
	const auto sz = multis.size();
	expansion<float> D;
	for (int i = tid; i < sz; i += KICK_BLOCK_SIZE) {
		const auto& check = ((tree*) multis[i]);
		array<float, NDIM> fpos;
		for (int dim = 0; dim < NDIM; dim++) {
			fpos[dim] = distance(pos[dim],check->pos[dim]);
		}
		flops += 6 + green_ewald(D, fpos);
		flops += multipole_interaction(L,check->multi, D);
		interacts++;
	}
	for (int i = 0; i < LP; i++) {
		Lreduce[tid] = L[i];
		cuda_sync();
		for (int P = KICK_BLOCK_SIZE / 2; P >= 1; P /= 2) {
			if (tid < P) {
				Lreduce[tid] += Lreduce[tid + P];
			}
			cuda_sync();
		}
		if (tid == 0) {
			params.L[params.depth][i] += Lreduce[0];
		}
	}
	kick_return_update_interactions_gpu(KR_EWCC, interacts, flops);
}

#endif

CUDA_DEVICE void cuda_cp_interactions(particle_set *parts, const vector<tree_ptr> &parti,
		kick_params_type *params_ptr) {
	kick_params_type &params = *params_ptr;
	const int &tid = threadIdx.x;
	volatile __shared__
	extern int shmem_ptr[];
	cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
	auto &Lreduce = shmem.Lreduce;
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
				const auto ip1 = i + 1;
				const auto other_tree = ((tree*) parti[ip1]);
				while (ip1 < partsz) {
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
		for (int i = 0; i < LP; i++) {
			Lreduce[tid] = L[i];
			cuda_sync();
			for (int P = KICK_BLOCK_SIZE / 2; P >= 1; P /= 2) {
				if (tid < P) {
					Lreduce[tid] += Lreduce[tid + P];
				}
				cuda_sync();
			}
			if (tid == 0) {
				params.L[params.depth][i] += Lreduce[0];
			}
		}
	}
	kick_return_update_interactions_gpu(KR_CP, interacts, flops);
}

CUDA_DEVICE void cuda_pp_interactions(particle_set *parts, const vector<tree_ptr> &parti,
		kick_params_type *params_ptr) {
	kick_params_type &params = *params_ptr;
	const int &tid = threadIdx.x;
	volatile __shared__
	extern int shmem_ptr[];
	cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
	auto &f = shmem.f;
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
		if (rungs[i] >= params.rung) {
			for (int dim = 0; dim < NDIM; dim++) {
				sinks[dim][i] = parts->pos(dim, i + myparts.first);
			}
		}
	}
	int i = 0;
	cuda_sync();
	auto these_parts = ((tree*) parti[0])->parts;
	const auto partsz = parti.size();
	while (i < partsz) {
		part_index = 0;
		while (part_index < KICK_PP_MAX && i < partsz) {
			const auto ip1 = i + 1;
			const auto other_tree = ((tree*) parti[ip1]);
			while (ip1 < partsz) {
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
		cuda_sync();
		array<float, NDIM> dx;
		auto &f0tid = f[0][tid];
		auto &f1tid = f[1][tid];
		auto &f2tid = f[2][tid];
		float& phi = f[NDIM][tid];
		auto &dx0 = dx[0];
		auto &dx1 = dx[1];
		auto &dx2 = dx[2];
		float r3inv, r1inv;
		for (int k = 0; k < nsinks; k++) {
			if (rungs[k] >= params.rung) {
				f0tid = 0.f;
				f1tid = 0.f;
				f2tid = 0.f;
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
						const float r1overh1 = sqrtf(r2) * hinv;              // 1 + FLOP_SQRT
						const float r2overh2 = r1overh1 * r1overh1;           // 1
						const float r3overh3 = r1overh1 * r2overh2;           // 1
						const float r5overh5 = r3overh3 * r1overh1;           // 1
						r1inv = fmaf(-0.3125f, r5overh5, 1.3125f * r3overh3 - fmaf(2.1875f, r1overh1, 2.1875f)); // 7
						r3inv = fmaf(r2overh2, (5.25f - 1.875f * r2overh2), PHI0); // 5
						flops += FLOP_SQRT + 16;
					}
					f0tid = fmaf(dx[0], r3inv, f0tid); // 2
					f1tid = fmaf(dx[1], r3inv, f1tid); // 2
					f2tid = fmaf(dx[2], r3inv, f2tid); // 2
					phi -= r1inv; // 1
					flops += 15;
					interacts++;
				}
				cuda_sync();
				for (int P = KICK_BLOCK_SIZE / 2; P >= 1; P /= 2) {
					if (tid < P) {
						for (int dim = 0; dim < NDIM + 1; dim++) {
							f[dim][tid] += f[dim][tid + P];
						}
					}
					cuda_sync();
				}
				if (tid == 0) {
					for (int dim = 0; dim < NDIM; dim++) {
						F[dim][k] -= f[dim][0];
					}
					Phi[k] += f[NDIM][0];
				}
			}
		}
	}
	kick_return_update_interactions_gpu(KR_PP, interacts, flops);
}

CUDA_DEVICE
void cuda_pc_interactions(particle_set *parts, const vector<tree_ptr> &multis, kick_params_type *params_ptr) {
	kick_params_type &params = *params_ptr;
	const int &tid = threadIdx.x;
	volatile
	__shared__
	extern int shmem_ptr[];
	cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
	auto &f = shmem.f;
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
		if (rungs[i] >= params.rung) {
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
	auto& f0tid = f[0][tid];
	auto& f1tid = f[1][tid];
	auto& f2tid = f[2][tid];
	auto& phi = f[NDIM][tid];
	auto& dx0 = dx[0];
	auto& dx1 = dx[1];
	auto& dx2 = dx[2];
	int nsrc = 0;
	const auto mmax = (((multis.size() - 1) / KICK_PC_MAX) + 1) * KICK_PC_MAX;
	for (int m = 0; m < mmax; m += KICK_PC_MAX) {
		nsrc = 0;
		for (int z = 0; z < KICK_PC_MAX; z++) {
			if (m + z < multis.size()) {
				const auto& other_ptr = ((tree*) multis[m + z]);
				const float* src = (float*) &other_ptr->multi;
				float* dst = (float*) &(msrcs[nsrc++]);
				if (tid < sizeof(multipole_pos) / sizeof(float)) {
					dst[tid] = src[tid];
				}
			}
		}
		cuda_sync();
		for (int k = 0; k < nparts; k++) {
			if (rungs[k] >= params.rung) {
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
				f0tid = Lforce[1];
				f1tid = Lforce[2];
				f2tid = Lforce[3];
				cuda_sync();
				for (int P = KICK_BLOCK_SIZE / 2; P >= 1; P /= 2) {
					if (tid < P) {
						for (int dim = 0; dim < NDIM + 1; dim++) {
							f[dim][tid] += f[dim][tid + P];
						}
					}
					cuda_sync();
				}
				if (tid == 0) {
					for (int dim = 0; dim < NDIM; dim++) {
						F[dim][k] -= f[dim][0];
					}
					Phi[k] += f[NDIM][0];
				}
			}
		}
	}
	kick_return_update_interactions_gpu(KR_PC, interacts, flops);
}

#ifdef TEST_FORCE
CUDA_KERNEL cuda_pp_ewald_interactions(particle_set *parts, size_t *test_parts, float *err, float *norm);

#ifdef __CUDA_ARCH__
CUDA_KERNEL cuda_pp_ewald_interactions(particle_set *parts, size_t *test_parts, float *err, float *norm) {
	const int &tid = threadIdx.x;
	const int &bid = blockIdx.x;
	ewald_const econst;

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
	for (int dim = 0; dim < NDIM; dim++) {
		f[dim][tid] = 0.0;
	}
	for (size_t source = tid; source < parts->size(); source += KICK_BLOCK_SIZE) {
		if (source != index) {
			array<float, NDIM> X;
			for (int dim = 0; dim < NDIM; dim++) {
				const auto a = sink[dim];
				const auto b = parts->pos(dim, source);
				X[dim] = distance(a, b);
			}
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
					const float d1 = (expfactor + erfc0) * r3inv;// 2
					for (int dim = 0; dim < NDIM; dim++) {
						f[dim][tid] -= dx[dim] * d1;
					}
				}
			}
			for (int i = 0; i < econst.nfour(); i++) {
				const auto &h = econst.four_index(i);
				const auto &hpart = econst.four_expansion(i);
				const float hdotx = h[0] * X[0] + h[1] * X[1] + h[2] * X[2];
				float so = sinf(2.0 * M_PI * hdotx);
				for (int dim = 0; dim < NDIM; dim++) {
					f[dim][tid] -= hpart(dim) * so;
				}
			}
		}
	}
	cuda_sync();
	for (int P = KICK_BLOCK_SIZE / 2; P >= 1; P /= 2) {
		if (tid < P) {
			for (int dim = 0; dim < NDIM; dim++) {
				f[dim][tid] += f[dim][tid + P];
			}
		}
		cuda_sync();
	}
	const auto f_ffm = sqrt(f_x * f_x + f_y * f_y + f_z * f_z);
	const auto f_dir = sqrt(sqr(f[0][0]) + sqr(f[1][0]) + sqr(f[2][0]));
	if (tid == 0) {
		norm[bid] = f_dir;
		err[bid] = fabsf(f_ffm - f_dir) / fabsf(f_dir);
	}
}
#endif

void cuda_compare_with_direct(particle_set *parts) {
	size_t *test_parts;
	float *errs;
	float *norms;
	CUDA_MALLOC(test_parts, N_TEST_PARTS);
	CUDA_MALLOC(errs, N_TEST_PARTS);
	CUDA_MALLOC(norms, N_TEST_PARTS);
	const size_t nparts = parts->size();
	for (int i = 0; i < N_TEST_PARTS; i++) {
		test_parts[i] = rand() % nparts;
	}
	cuda_pp_ewald_interactions<<<N_TEST_PARTS,KICK_BLOCK_SIZE>>>(parts, test_parts, errs, norms);
	CUDA_CHECK(cudaDeviceSynchronize());
	float avg_err = 0.0;
	float norm = 0.0;
	float err_max = 0.0;
	for (int i = 0; i < N_TEST_PARTS; i++) {
		avg_err += errs[i];
		err_max = fmaxf(err_max, errs[i]);
		norm += norms[i];
	}
	avg_err /= N_TEST_PARTS;
//	err_max /= (norm / N_TEST_PARTS);
//   avg_err /= norm;
	printf("Avg Error is %e\n", avg_err);
	printf("Max Error is %e\n", err_max);
	CUDA_FREE(norms);
	CUDA_FREE(errs);
	CUDA_FREE(test_parts);
}

#endif
