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
#include <cosmictiger/tree.hpp>

#define MIN_KICK_WARP 10

extern __constant__ kick_constants constant;

CUDA_DEVICE void cuda_cc_interactions(kick_params_type *params_ptr, eval_type etype) {
	kick_params_type &params = *params_ptr;
	__shared__
	extern int shmem_ptr[];
	cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
	const auto& multis = shmem.multi_interactions;
	const int &tid = threadIdx.x;
//	auto &Lreduce = shmem.Lreduce;
	if (multis.size() == 0) {
		return;
	}
	expansion<float> L; // = shmem.expanse2[tid];
	expansion<float> D; // = shmem.expanse1[tid];
	int flops = 0;
	int interacts = 0;
	for (int i = 0; i < LP; i++) {
		L[i] = 0.0f;
	}
	const auto pos = shmem.self.get_pos();
	const int sz = multis.size();
	for (int i = tid; i < sz; i += warpSize) {
		const multipole mpole = multis[i].get_multi();
		array<float, NDIM> fpos;
		const auto other_pos = multis[i].get_pos();
		for (int dim = 0; dim < NDIM; dim++) {
			fpos[dim] = distance(pos[dim], other_pos[dim]);
		}
		flops += 6;
		if (etype == DIRECT) {
			flops += green_direct(D, fpos);
		} else {
			flops += green_ewald(D, fpos);
		}
		flops += multipole_interaction(L, mpole, D, constant.full_eval);
		interacts++;
	}

	for (int P = warpSize / 2; P >= 1; P /= 2) {
		for (int i = 0; i < LP; i++) {
			L[i] += __shfl_xor_sync(0xffffffff, L[i], P);
		}
	}
	for (int i = tid; i < LP; i += warpSize) {
		NAN_TEST(L[i]);
		params.L[shmem.depth][i] += L[i];
	}
	__syncwarp();
	if (constant.full_eval) {
		kick_return_update_interactions_gpu(etype == DIRECT ? KR_CC : KR_EWCC, interacts, flops);
	}
}

CUDA_DEVICE void cuda_cp_interactions(kick_params_type *params_ptr) {
	particle_set& parts = *(particle_set*) constant.particles;
	kick_params_type &params = *params_ptr;
	__shared__
	extern int shmem_ptr[];
	cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
	const auto& parti = shmem.part_interactions;
	const int &tid = threadIdx.x;
	//auto &Lreduce = shmem.Lreduce;
	if (parti.size() == 0) {
		return;
	}
	auto &sources = shmem.src;
	const auto myparts = shmem.self.get_parts();
	int part_index;
	int flops = 0;
	int interacts = 0;
	expansion<float> L;
	if (parti.size() > 0) {
		for (int j = 0; j < LP; j++) {
			L[j] = 0.0;
		}
		auto these_parts = parti[0].get_parts();
		int i = 0;
		const auto pos = shmem.self.get_pos();
		const auto partsz = parti.size();
		while (i < partsz) {
			part_index = 0;
			while (part_index < KICK_PP_MAX && i < partsz) {
				while (i + 1 < partsz) {
					const auto other_tree_parts = parti[i + 1].get_parts();
					if (these_parts.second == other_tree_parts.first) {
						these_parts.second = other_tree_parts.second;
						i++;
					} else {
						break;
					}
				}
				const part_int imin = these_parts.first;
				const part_int imax = min(these_parts.first + (KICK_PP_MAX - part_index), these_parts.second);
				const int sz = imax - imin;
				for (int j = tid; j < sz; j += warpSize) {
					for (int dim = 0; dim < NDIM; dim++) {
						sources[dim][part_index + j] = parts.pos(dim, j + imin);
					}
				}
				these_parts.first += sz;
				part_index += sz;
				if (these_parts.first == these_parts.second) {
					i++;
					if (i < parti.size()) {
						these_parts = parti[i].get_parts();
					}
				}
			}
			for (int j = tid; j < part_index; j += warpSize) {
				array<float, NDIM> dx;
				dx[0] = distance(pos[0], sources[0][j]);
				dx[1] = distance(pos[1], sources[1][j]);
				dx[2] = distance(pos[2], sources[2][j]);
				expansion<float> D;
				flops += 3;
				flops += green_direct(D, dx);
				flops += multipole_interaction(L, D);
				interacts++;
			}
		}
		for (int P = warpSize / 2; P >= 1; P /= 2) {
			for (int i = 0; i < LP; i++) {
				L[i] += __shfl_xor_sync(0xffffffff, L[i], P);
			}
		}
		for (int i = tid; i < LP; i += warpSize) {
			NAN_TEST(L[i]);
			params.L[shmem.depth][i] += L[i];
		}
	}
	__syncwarp();
	if (constant.full_eval) {
		kick_return_update_interactions_gpu(KR_CP, interacts, flops);
	}
}

CUDA_DEVICE int compress_sinks(kick_params_type *params_ptr) {
	particle_set& parts = *(particle_set*) constant.particles;
	const int &tid = threadIdx.x;
	__shared__
	extern int shmem_ptr[];
	cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
	auto &sinks = shmem.sink;
	auto& act_map = shmem.act_map;

	const auto myparts = shmem.self.get_parts();
	const int nsinks = myparts.second - myparts.first;
	const int nsinks_max = max(((nsinks - 1) / warpSize + 1) * warpSize, 0);

	int my_index;
	bool found;
	int base = 0;
	int nactive = 0;
	int total;

	for (int i = tid; i < nsinks_max; i += warpSize) {
		my_index = 0;
		found = false;
		if (i < nsinks) {
			if (parts.rung(i + myparts.first) >= constant.rung || constant.full_eval) {
				found = true;
				my_index = 1;
				nactive++;
			}
		}
		int tmp;
		for (int P = 1; P < warpSize; P *= 2) {
			tmp = __shfl_up_sync(0xFFFFFFFF, my_index, P);
			if (tid >= P) {
				my_index += tmp;
			}
		}
		total = __shfl_sync(0xFFFFFFFF, my_index, warpSize - 1);
		tmp = __shfl_up_sync(0xFFFFFFFF, my_index, 1);
		if (tid > 0) {
			my_index = tmp;
		} else {
			my_index = 0;
		}
		if (found) {
			act_map[base + my_index] = i;
			//		PRINT( "%i %i\n", base+my_index, i);
		}
		base += total;
	}
	for (int P = warpSize / 2; P >= 1; P /= 2) {
		nactive += __shfl_xor_sync(0xFFFFFFFF, nactive, P);
	}
	for (int i = tid; i < nactive; i += warpSize) {
		for (int dim = 0; dim < NDIM; dim++) {
			sinks[dim][i] = parts.pos(dim, act_map[i] + myparts.first);
		}
	}
	return nactive;
}

CUDA_DEVICE void cuda_pp_interactions(kick_params_type *params_ptr, int nactive) {
	particle_set& parts = *(particle_set*) constant.particles;
	kick_params_type &params = *params_ptr;
	const int &tid = threadIdx.x;
	__shared__
	extern int shmem_ptr[];
	cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
	const auto& parti = shmem.part_interactions;
	auto &F = params.F;
	auto &Phi = params.Phi;
	auto &sources = shmem.src;
	auto &sinks = shmem.sink;
	auto& act_map = shmem.act_map;
	const auto& h2 = constant.h2;
	const auto h3inv = constant.h3;
	const auto& hinv = constant.hinv;
	int flops = 0;
	int interacts = 0;
	int part_index;
	if (parti.size() == 0) {
		return;
	}
//   PRINT( "%i\n", parti.size());
	const auto myparts = shmem.self.get_parts();
	int i = 0;
	auto these_parts = parti[0].get_parts();

	const auto partsz = parti.size();
	while (i < partsz) {
		part_index = 0;
		while (part_index < KICK_PP_MAX && i < partsz) {
			while (i + 1 < partsz) {
				const auto other_tree_parts = parti[i + 1].get_parts();
				if (these_parts.second == other_tree_parts.first) {
					these_parts.second = other_tree_parts.second;
					i++;
				} else {
					break;
				}
			}
			const part_int& imin = these_parts.first;
			const part_int imax = min(these_parts.first + (KICK_PP_MAX - part_index), these_parts.second);
			const int sz = imax - imin;
			for (int j = tid; j < sz; j += warpSize) {
				for (int dim = 0; dim < NDIM; dim++) {
					sources[dim][part_index + j] = parts.pos(dim, j + imin);
				}
			}
			these_parts.first += sz;
			part_index += sz;
			if (these_parts.first == these_parts.second) {
				i++;
				if (i < partsz) {
					these_parts = parti[i].get_parts();
				}
			}
		}
		float fx;
		float fy;
		float fz;
		float phi;
		float dx0, dx1, dx2;
		float r3inv, r1inv;
		__syncwarp();
		for (int k = tid; k < nactive; k += warpSize) {
			fx = 0.f;
			fy = 0.f;
			fz = 0.f;
			phi = 0.f;
			for (int j = 0; j < part_index; j++) {
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
					if (constant.full_eval) {
						r1inv = -5.0f / 16.0f;
						r1inv = fmaf(r1inv, r2oh2, 21.0f / 16.0f);
						r1inv = fmaf(r1inv, r2oh2, -35.0f / 16.0f);
						r1inv = fmaf(r1inv, r2oh2, 35.0f / 16.0f);
					}
					r1inv *= hinv;
					r3inv *= h3inv;
					flops += FLOP_SQRT + 18;
				}
				fx = fmaf(dx0, r3inv, fx); // 2
				fy = fmaf(dx1, r3inv, fy); // 2
				fz = fmaf(dx2, r3inv, fz); // 2
				NAN_TEST(fx);NAN_TEST(fy);NAN_TEST(fz);
				phi -= r1inv; // 1
				flops += 15;
				interacts++;
			}
			const auto l = act_map[k];
			F[0][l] -= fx;
			F[1][l] -= fy;
			F[2][l] -= fz;
			if (constant.full_eval) {
				Phi[l] += phi;
			}
		}
	}
	__syncwarp();
	if (constant.full_eval) {
		kick_return_update_interactions_gpu(KR_PP, interacts, flops);
	}
}

CUDA_DEVICE
void cuda_pc_interactions(kick_params_type *params_ptr, int nactive) {
	kick_params_type &params = *params_ptr;
	const int &tid = threadIdx.x;

	__shared__
	extern int shmem_ptr[];
	cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
	const auto& multis = shmem.multi_interactions;
	auto &F = params.F;
	auto& act_map = shmem.act_map;
	auto &Phi = params.Phi;
	const auto myparts = shmem.self.get_parts();
	if (multis.size() == 0) {
		return;
	}
	auto &sinks = shmem.sink;
	auto& msrcs = shmem.msrc;

	int interacts = 0;
	int flops = 0;
	array<float, NDIM> dx;
	tensor_trless_sym<float, 2> Lforce;
	expansion<float> D;
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
				const float* src = multis[m + z].get_multi_ptr();
				float* dst = (float*) &(msrcs[nsrc]);
				nsrc++;
				for (int k = tid; k < msize; k += KICK_BLOCK_SIZE) {
					dst[tid] = __ldg(src + tid);
				}
			}
		}
		__syncwarp();
		for (int k = tid; k < nactive; k += warpSize) {
			Lforce = 0.0;
			for (int i = 0; i < nsrc; i++) {
				const auto &source = msrcs[i].pos;
				dx0 = distance(sinks[0][k], source[0]);
				dx1 = distance(sinks[1][k], source[1]);
				dx2 = distance(sinks[2][k], source[2]);
				flops += 6;
				flops += green_direct(D, dx);
				flops += multipole_interaction(Lforce, msrcs[i].multi, D, constant.full_eval);
				interacts++;
			}
			const int l = act_map[k];
			F[0][l] -= Lforce(1, 0, 0);
			F[1][l] -= Lforce(0, 1, 0);
			F[2][l] -= Lforce(0, 0, 1);
			if (constant.full_eval) {
				Phi[l] += Lforce(0, 0, 0);
			}
		}
		__syncwarp();
	}
	__syncwarp();
	if (constant.full_eval) {
		kick_return_update_interactions_gpu(KR_PC, interacts, flops);
	}

}

CUDA_KERNEL cuda_pp_ewald_interactions(particle_set *parts, part_int *test_parts, float *ferr, float *fnorm,
		float* perr, float* pnorm, float GM, float h);
