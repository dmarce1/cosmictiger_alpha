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
	L.scale_back();
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
	kick_params_type &params = *params_ptr;
	auto* part_sets = (particle_sets*) constant.partsets;
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
	part_iters these_parts;
	for (int pi = 0; pi < constant.npart_types; pi++) {
		for (int j = 0; j < LP; j++) {
			L[j] = 0.0;
		}

		float mass = part_sets->weights[pi];
		auto& parts = *part_sets->sets[pi];
		these_parts = parti[0].get_parts(pi);
		int i = 0;
		const auto pos = shmem.self.get_pos();
		const auto partsz = parti.size();
		while (i < partsz) {
			part_index = 0;
			while (part_index < KICK_PP_MAX && i < partsz) {
				while (i + 1 < partsz) {
					part_iters other_tree_parts;
					other_tree_parts = parti[i + 1].get_parts(pi);
					if (these_parts.second == other_tree_parts.first) {
						these_parts.second = other_tree_parts.second;
						i++;
					} else {
						break;
					}
				}
				const size_t imin = these_parts.first;
				const size_t imax = min(these_parts.first + (KICK_PP_MAX - part_index), these_parts.second);
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
						these_parts = parti[i].get_parts(pi);
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
			params.L[shmem.depth][i] += mass * L[i];
		}
	}
	__syncwarp();
	if (constant.full_eval) {
		kick_return_update_interactions_gpu(KR_CP, interacts, flops);
	}
}

CUDA_DEVICE int compress_sinks(kick_params_type *params_ptr) {
	const int &tid = threadIdx.x;
	auto* part_sets = (particle_sets*) constant.partsets;
	__shared__
	extern int shmem_ptr[];
	cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
	auto &sinks = shmem.sink;
	auto& act_map = shmem.act_map;

	parts_type myparts = shmem.self.get_parts();

	int my_index;
	bool found;
	int base = 0;
	int nactive;
	int tot_nactive = 0;
	int total;

	for (int pi = 0; pi < constant.npart_types; pi++) {
		nactive = 0;
		const int nsinks = myparts[pi].second - myparts[pi].first;
		const int nsinks_max = round_up(nsinks, warpSize);
		auto& parts = *part_sets->sets[pi];
		const int offset = myparts.index_offset(pi);
		for (int i = tid; i < nsinks_max; i += warpSize) {
			my_index = 0;
			found = false;
			if (i < nsinks) {
				if (parts.rung(i + myparts[pi].first) >= constant.rung || constant.full_eval) {
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
				act_map[base + my_index] = i + offset;
			}
			base += total;
		}
		for (int P = warpSize / 2; P >= 1; P /= 2) {
			nactive += __shfl_xor_sync(0xFFFFFFFF, nactive, P);
		}
		for (int i = tid; i < nactive; i += warpSize) {
			for (int dim = 0; dim < NDIM; dim++) {
				sinks[dim][i + tot_nactive] = parts.pos(dim, act_map[i + tot_nactive] + myparts[pi].first - offset);
			}
		}
		tot_nactive += nactive;
	}
	return tot_nactive;
}

CUDA_DEVICE void cuda_pp_interactions(kick_params_type *params_ptr, int nactive) {
	kick_params_type &params = *params_ptr;
	auto* part_sets = (particle_sets*) constant.partsets;
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
	const auto& hinv = constant.hinv;
	int flops = 0;
	int interacts = 0;
	int part_index;
	if (parti.size() == 0) {
		return;
	}
	for (int pi = 0; pi < constant.npart_types; pi++) {
		auto& parts = *part_sets->sets[pi];
		float mass = part_sets->weights[pi];
		int i = 0;
		part_iters these_parts;
		these_parts = parti[0].get_parts(pi);
		const auto partsz = parti.size();
		while (i < partsz) {
			part_index = 0;
			while (part_index < KICK_PP_MAX && i < partsz) {
				while (i + 1 < partsz) {
					part_iters other_tree_parts;
					other_tree_parts = parti[i + 1].get_parts(pi);
					if (these_parts.second == other_tree_parts.first) {
						these_parts.second = other_tree_parts.second;
						i++;
					} else {
						break;
					}
				}
				const size_t imin = these_parts.first;
				const size_t imax = min(these_parts.first + (KICK_PP_MAX - part_index), these_parts.second);
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
						these_parts = parti[i].get_parts(pi);
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
			int kmid;
			if ((nactive % warpSize) < MIN_KICK_WARP) {
				kmid = nactive - (nactive % warpSize);
			} else {
				kmid = nactive;
			}
			for (int k = tid; k < kmid; k += warpSize) {
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
						flops += FLOP_SQRT + 16;
					}
					const float massr3inv = mass * r3inv;
					fx = fmaf(dx0, massr3inv, fx); // 2
					fy = fmaf(dx1, massr3inv, fy); // 2
					fz = fmaf(dx2, massr3inv, fz); // 2
					NAN_TEST(fx);NAN_TEST(fy);NAN_TEST(fz);
					phi -= r1inv * mass; // 1
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
//			}
			}
			__syncwarp();
			for (int k = kmid; k < nactive; k++) {
				fx = 0.f;
				fy = 0.f;
				fz = 0.f;
				phi = 0.f;
				for (int j = tid; j < part_index; j += warpSize) {
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
						flops += FLOP_SQRT + 16;
					}
					const float massr3inv = mass * r3inv;
					fx = fmaf(dx0, massr3inv, fx); // 2
					fy = fmaf(dx1, massr3inv, fy); // 2
					fz = fmaf(dx2, massr3inv, fz); // 2
					NAN_TEST(fx);NAN_TEST(fy);NAN_TEST(fz);
					phi -= r1inv * mass; // 1
					flops += 15;
					interacts++;
				}
				for (int P = warpSize / 2; P >= 1; P /= 2) {
					fx += __shfl_down_sync(0xffffffff, fx, P);
					fy += __shfl_down_sync(0xffffffff, fy, P);
					fz += __shfl_down_sync(0xffffffff, fz, P);
					if (constant.full_eval) {
						phi += __shfl_down_sync(0xffffffff, phi, P);
					}
				}
				if (tid == 0) {
					const auto l = act_map[k];
					F[0][l] -= fx;
					F[1][l] -= fy;
					F[2][l] -= fz;
					if (constant.full_eval) {
						Phi[l] += phi;
					}
				}
			}
		}
		__syncwarp();
	}
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
	array<float, NDIM + 1> Lforce;
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
				if (tid < msize) {
					dst[tid] = __ldg(src + tid);
				}
			}
		}
		__syncwarp();
		int kmid;
		if ((nactive % warpSize) < MIN_KICK_WARP) {
			kmid = nactive - (nactive % warpSize);
		} else {
			kmid = nactive;
		}
		for (int k = tid; k < kmid; k += warpSize) {
			for (int i = 0; i < NDIM + 1; i++) {
				Lforce[i] = 0.f;
			}
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
			F[0][l] -= Lforce[1];
			F[1][l] -= Lforce[2];
			F[2][l] -= Lforce[3];
			if (constant.full_eval) {
				Phi[l] += Lforce[0];
			}
		}
		__syncwarp();
		for (int k = kmid; k < nactive; k++) {
			for (int i = 0; i < NDIM + 1; i++) {
				Lforce[i] = 0.f;
			}
			for (int i = tid; i < nsrc; i += warpSize) {
				const auto &source = msrcs[i].pos;
				dx0 = distance(sinks[0][k], source[0]);
				dx1 = distance(sinks[1][k], source[1]);
				dx2 = distance(sinks[2][k], source[2]);
				flops += 6;
				flops += green_direct(D, dx);
				flops += multipole_interaction(Lforce, msrcs[i].multi, D, constant.full_eval);
				interacts++;
			}
			for (int P = warpSize / 2; P >= 1; P /= 2) {
				for (int dim = 0; dim < NDIM + 1; dim++) {
					Lforce[dim] += __shfl_down_sync(0xffffffff, Lforce[dim], P);
				}
			}
			if (tid == 0) {
				const int l = act_map[k];
				F[0][l] -= Lforce[1];
				F[1][l] -= Lforce[2];
				F[2][l] -= Lforce[3];
				if (constant.full_eval) {
					Phi[l] += Lforce[0];
				}
			}
		}
	}
	__syncwarp();
	if (constant.full_eval) {
		kick_return_update_interactions_gpu(KR_PC, interacts, flops);
	}

}

CUDA_KERNEL cuda_pp_ewald_interactions(particle_set *parts, size_t *test_parts, float *ferr, float *fnorm, float* perr,
		float* pnorm, float GM, float h);

#ifdef __CUDA_ARCH__
CUDA_KERNEL cuda_pp_ewald_interactions(particle_set *parts, size_t *test_parts, float *ferr, float *fnorm, float* perr, float* pnorm, float GM, float h) {
	const int &tid = threadIdx.x;
	const int &bid = blockIdx.x;
	const auto hinv = 1.f / h;
	const auto h2 = h * h;
	const auto index = test_parts[bid];
	array<fixed32,NDIM> sink;
	sink[0] = parts->pos(0,index);
	sink[1] = parts->pos(1,index);
	sink[2] = parts->pos(2,index);
	const auto f_x = parts->force(0, index);
	const auto f_y = parts->force(1, index);
	const auto f_z = parts->force(2, index);
	__shared__ array<array<double, EWALD_BLOCK_SIZE>, NDIM>
	f;
	__shared__ array<double,EWALD_BLOCK_SIZE> phi;
	for (int dim = 0; dim < NDIM; dim++) {
		f[dim][tid] = 0.0;
	}
	phi[tid] = 0.0;
	for (size_t source = tid; source < parts->size(); source += EWALD_BLOCK_SIZE) {
		if (source != index) {
			array<float, NDIM> X;
			for (int dim = 0; dim < NDIM; dim++) {
				const auto a = sink[dim];
				const auto b = parts->pos(dim, source);
				X[dim] = distance(a, b);
			}
			const auto r2 = fmaf(X[0],X[0],fmaf(X[1],X[1],sqr(X[2])));
			if (r2 < h2) {
				const float r1oh1 = sqrtf(r2) * hinv;              // 1 + FLOP_SQRT
				const float r2oh2 = r1oh1 * r1oh1;// 1
				float r3inv = +15.0f / 8.0f;
				r3inv = fmaf(r3inv, r2oh2, -21.0f / 4.0f);
				r3inv = fmaf(r3inv, r2oh2, +35.0f / 8.0f);
				float r1inv = -5.0f / 16.0f;
				r1inv = fmaf(r1inv, r2oh2, 21.0f / 16.0f);
				r1inv = fmaf(r1inv, r2oh2, -35.0f / 16.0f);
				r1inv = fmaf(r1inv, r2oh2, 35.0f / 16.0f);
				phi[tid] -= r1inv;
				for( int dim = 0; dim < NDIM; dim++) {
					f[dim][tid] -= X[dim] * r3inv;
				}
			} else {
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
			}
		}
	}
	__syncthreads();
	for (int P = EWALD_BLOCK_SIZE / 2; P >= 1; P /= 2) {
		if (tid < P) {
			for (int dim = 0; dim < NDIM; dim++) {
				f[dim][tid] += f[dim][tid + P];
			}
			phi[tid] += phi[tid + P];
		}
		__syncthreads();
	}
	for(int dim = 0; dim < NDIM; dim++) {
		f[dim][0] *= GM;
	}
	phi[0] *= GM;
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
	cuda_pp_ewald_interactions<<<N_TEST_PARTS,EWALD_BLOCK_SIZE>>>(parts, test_parts, ferrs, fnorms, perrs, pnorms, global().opts.G * global().opts.M, global().opts.hsoft);
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

