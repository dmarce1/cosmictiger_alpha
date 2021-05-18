#include <cosmictiger/gravity.hpp>
#include <cosmictiger/kick_return.hpp>
#include <cosmictiger/particle_server.hpp>

void tree::cpu_cc_direct(kick_params_type *params_ptr) {
	kick_params_type &params = *params_ptr;
	auto &L = params.L[params.depth];
	auto &multis = params.multi_interactions;
	int flops = 0;
	int interacts = 0;
	array<simd_int, NDIM> X;
	array<simd_int, NDIM> Y;
	array<simd_float, NDIM> dX;
	expansion<simd_float> D;
	multipole_type<simd_float> M;
	expansion<simd_float> Lacc;
	const auto pos = params.tptr.get_pos();
	for (int dim = 0; dim < NDIM; dim++) {
		X[dim] = fixed<int>(pos[dim]).raw();
	}
	if (multis.size()) {
		for (int i = 0; i < LP; i++) {
			Lacc[i] = 0.f;
		}
		const auto cnt1 = multis.size();
		for (int j = 0; j < cnt1; j += simd_float::size()) {
			int n = 0;
			for (int k = 0; k < simd_float::size(); k++) {
				if (j + k < cnt1) {
					const multipole mpole = multis[j + k].get_multi();
					const auto other_pos = multis[j + k].get_pos();
					for (int dim = 0; dim < NDIM; dim++) {
						Y[dim][k] = fixed<int>(other_pos[dim]).raw();
					}
					for (int i = 0; i < MP; i++) {
						M[i][k] = mpole[i];
					}
					n++;
				} else {
					const auto other_pos = multis[cnt1 - 1].get_pos();
					for (int dim = 0; dim < NDIM; dim++) {
						Y[dim][k] = fixed<int>(other_pos[dim]).raw();
					}
					for (int i = 0; i < MP; i++) {
						M[i][k] = 0.f;
					}
				}
			}
			interacts += n;
			for (int dim = 0; dim < NDIM; dim++) {
				dX[dim] = distance(X[dim], Y[dim]);
			}
			flops += n * 6;
			flops += n * green_direct(D, dX);
			flops += n * multipole_interaction(Lacc, M, D, params.full_eval);
		}
		Lacc.scale_back();
		for (int k = 0; k < simd_float::size(); k++) {
			for (int i = 0; i < LP; i++) {
				NAN_TEST(Lacc[i][k]);
				L[i] += Lacc[i][k];
			}
		}
	}
	if (params.full_eval) {
		kick_return_update_interactions_cpu(KR_CC, interacts, flops);
	}
}

void tree::cpu_cp_direct(kick_params_type *params_ptr) {
	particle_server pserv;
	kick_params_type &params = *params_ptr;
	auto &L = params.L[params.depth];
	const auto parts = params.tptr.get_parts();
	int nparts = parts.second - parts.first;
	int flops = 0;
	int interacts = 0;
	int n;
	static thread_local std::array<std::vector<fixed32>, NDIM> sources;
	auto &partis = params.part_interactions;
	for (int dim = 0; dim < NDIM; dim++) {
		sources[dim].resize(0);
	}
	for (int k = 0; k < partis.size(); k++) {
		const auto& other_parts = partis[k].get_parts();
		pserv.read_positions(sources, partis[k].rank, other_parts);
	}
	array<simd_int, NDIM> X;
	array<simd_int, NDIM> Y;
	array<simd_float, NDIM> dX;
	expansion<simd_float> D;
	simd_float M;
	expansion<simd_float> Lacc;
	const auto pos = params.tptr.get_pos();
	for (int dim = 0; dim < NDIM; dim++) {
		X[dim] = fixed<int>(pos[dim]).raw();
	}
	for (int i = 0; i < LP; i++) {
		Lacc[i] = 0.f;
	}
	const auto cnt1 = sources[0].size();
	for (int j = 0; j < cnt1; j += simd_float::size()) {
		n = 0;
		for (int k = 0; k < simd_float::size(); k++) {
			if (j + k < cnt1) {
				for (int dim = 0; dim < NDIM; dim++) {
					Y[dim][k] = sources[dim][j + k].raw();
				}
				for (int i = 0; i < MP; i++) {
					M[k] = 1.f;
				}
				n++;
			} else {
				for (int dim = 0; dim < NDIM; dim++) {
					Y[dim][k] = sources[dim][cnt1 - 1].raw();
				}
				for (int i = 0; i < MP; i++) {
					M[k] = 0.f;
				}
			}
		}
		for (int dim = 0; dim < NDIM; dim++) {
			dX[dim] = distance(X[dim], Y[dim]);
		}
		flops += n * 6;
		flops += n * green_direct(D, dX);
		flops += n * multipole_interaction(Lacc, M, D);
	}
	Lacc.scale_back();
	for (int k = 0; k < simd_float::size(); k++) {
		for (int i = 0; i < LP; i++) {
			L[i] += Lacc[i][k];
		}
	}
	if (params.full_eval) {
		kick_return_update_interactions_cpu(KR_CP, interacts, flops);
	}
}

void tree::cpu_pp_direct(kick_params_type *params_ptr) {
	particle_server pserv;
	auto* particles = &pserv.get_particle_set();
	kick_params_type &params = *params_ptr;
	auto &L = params.L[params.depth];
	auto& F = params.F;
	auto& Phi = params.Phi;
	const simd_float h2(params.hsoft * params.hsoft);
	const simd_float hinv(1.f / params.hsoft);
	const simd_float tiny(1.0e-20);
	int flops = 0;
	int interacts = 0;
	int n;
	const auto parts = params.tptr.get_parts();
	int nparts = parts.second - parts.first;
	static thread_local std::array<std::vector<fixed32>, NDIM> sources;
	auto &partis = params.part_interactions;
	for (int dim = 0; dim < NDIM; dim++) {
		sources[dim].resize(0);
	}
	for (int k = 0; k < partis.size(); k++) {
		const auto& other_parts = partis[k].get_parts();
		pserv.read_positions(sources, partis[k].rank, other_parts);
	}
	simd_float mask;
	array<simd_int, NDIM> X;
	array<simd_int, NDIM> Y;
	for (int i = 0; i < nparts; i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			X[dim] = particles->pos(dim, i + parts.first).raw();
		}
		if (particles->rung(i + parts.first) >= params.rung || params.full_eval) {
			array<simd_float, NDIM> f;
			simd_float phi = 0.f;
			for (int dim = 0; dim < NDIM; dim++) {
				f[dim] = simd_float(0.f);
			}
			for (int j = 0; j < sources[0].size(); j += simd_float::size()) {
				n = 0;
				for (int k = 0; k < simd_float::size(); k++) {
					if (j + k < sources[0].size()) {
						mask[k] = 1.f;
						for (int dim = 0; dim < NDIM; dim++) {
							Y[dim][k] = sources[dim][j + k].raw();
						}
						interacts++;
						n++;
					} else {
						mask[k] = 0.f;
						for (int dim = 0; dim < NDIM; dim++) {
							Y[dim][k] = sources[dim][sources[0].size() - 1].raw();
						}
					}
				}
				array<simd_float, NDIM> dX;
				for (int dim = 0; dim < NDIM; dim++) {
					dX[dim] = distance(X[dim], Y[dim]);
				}
				const simd_float r2 = max(fma(dX[0], dX[0], fma(dX[1], dX[1], dX[2] * dX[2])), tiny);                   // 6
				const simd_float far_flag = r2 > h2;                                                                    // 1
				simd_float rinv1, rinv3;
				if (far_flag.sum() == simd_float::size()) {                                                             // 7
					rinv1 = mask * simd_float(1) / sqrt(r2);                                      // 1 + FLOP_DIV + FLOP_SQRT
					rinv3 = rinv1 * rinv1 * rinv1;                                                     // 2
					flops += n * (2 + FLOP_DIV + FLOP_SQRT);
				} else {
					const simd_float rinv1_far = mask * simd_float(1) / sqrt(r2);                 // 1 + FLOP_DIV + FLOP_SQRT
					const simd_float rinv3_far = rinv1_far * rinv1_far * rinv1_far;                                      // 2
					const simd_float r1overh1 = sqrt(r2) * hinv;                                             // FLOP_SQRT + 1
					const simd_float r2oh2 = r1overh1 * r1overh1;                                                     // 1
					simd_float rinv3_near = +15.0f / 8.0f;
					rinv3_near = fma(rinv3_near, r2oh2, simd_float(-21.0f / 4.0f));
					rinv3_near = fma(rinv3_near, r2oh2, simd_float(+35.0f / 8.0f));
					simd_float rinv1_near = -5.0f / 16.0f;
					rinv1_near = fma(rinv1_near, r2oh2, simd_float(21.0f / 16.0f));
					rinv1_near = fma(rinv1_near, r2oh2, simd_float(-35.0f / 16.0f));
					rinv1_near = fma(rinv1_near, r2oh2, simd_float(35.0f / 16.0f));
					rinv1 = far_flag * rinv1_far + (simd_float(1) - far_flag) * rinv1_near * mask;                  // 4
					rinv3 = far_flag * rinv3_far + (simd_float(1) - far_flag) * rinv3_near * mask;                  // 4
					flops += n * (28 + FLOP_DIV + 2 * FLOP_SQRT);
				}
				for (int dim = 0; dim < NDIM; dim++) {
					f[dim] = fma(rinv3, dX[dim], f[dim]);                                                                // 2
				}
				phi -= rinv1;                                                                                           // 1
				flops += n * (27 + 2 * FLOP_SQRT + FLOP_DIV);
			}
			for (int dim = 0; dim < NDIM; dim++) {
				F[dim][i] -= f[dim].sum();
			}
			Phi[i] += phi.sum();
		}
	}
	if (params.full_eval) {
		kick_return_update_interactions_cpu(KR_PP, interacts, flops);
	}
}

void tree::cpu_pc_direct(kick_params_type *params_ptr) {
	particle_server pserv;
	auto* particles = &pserv.get_particle_set();
	kick_params_type &params = *params_ptr;
	auto &L = params.L[params.depth];
	auto &multis = params.multi_interactions;
	auto& F = params.F;
	auto& Phi = params.Phi;
	const auto parts = params.tptr.get_parts();
	int nparts = parts.second - parts.first;
	array<simd_int, NDIM> X;
	array<simd_float, NDIM> dX;
	array<simd_int, NDIM> Y;
	multipole_type<simd_float> M;
	expansion<simd_float> D;
	int flops = 0;
	int n;
	int interacts = 0;
	for (int i = 0; i < nparts; i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			X[dim] = particles->pos(dim, i + parts.first).raw();
		}
		if (particles->rung(i + parts.first) >= params.rung || params.full_eval) {
			array<simd_float, NDIM> f;
			array<simd_float, NDIM + 1> Lacc;
			const auto cnt1 = multis.size();
			for (int j = 0; j < NDIM + 1; j++) {
				Lacc[j] = 0.f;
			}
			for (int j = 0; j < cnt1; j += simd_float::size()) {
				n = 0;
				for (int k = 0; k < simd_float::size(); k++) {
					if (j + k < cnt1) {
						const multipole mpole = multis[j + k].get_multi();
						const auto other_pos = multis[j + k].get_pos();
						for (int dim = 0; dim < NDIM; dim++) {
							Y[dim][k] = fixed<int>(other_pos[dim]).raw();
						}
						for (int i = 0; i < MP; i++) {
							M[i][k] = mpole[i];
						}
						n++;
					} else {
						const auto other_pos = multis[cnt1 - 1].get_pos();
						for (int dim = 0; dim < NDIM; dim++) {
							Y[dim][k] = fixed<int>(other_pos[dim]).raw();
						}
						for (int i = 0; i < MP; i++) {
							M[i][k] = 0.f;
						}
					}
				}
				for (int dim = 0; dim < NDIM; dim++) {
					dX[dim] = distance(X[dim], Y[dim]);
				}
				flops += n * 6;
				flops += n * green_direct(D, dX);
				flops += n * multipole_interaction(Lacc, M, D, params.full_eval);
			}
			Phi[i] += Lacc[0].sum();                                                                           // * DSCALE;
			for (int dim = 0; dim < NDIM; dim++) {
				F[dim][i] -= Lacc[1 + dim].sum();                                                    // * (DSCALE * DSCALE);
			}
		}
	}
	if (params.full_eval) {
		kick_return_update_interactions_cpu(KR_PC, interacts, flops);
	}
}

void tree::cpu_cc_ewald(kick_params_type *params_ptr) {
#ifdef PERIODIC_OFF
	return;
#endif
	kick_params_type &params = *params_ptr;
	auto &L = params.L[params.depth];
	auto &multis = params.multi_interactions;
	int flops = 0;
	int interacts = 0;
	array<simd_int, NDIM> X;
	array<simd_int, NDIM> Y;
	array<simd_float, NDIM> dX;
	expansion<simd_float> D;
	multipole_type<simd_float> M;
	expansion<simd_float> Lacc;
	const auto pos = params.tptr.get_pos();
	for (int dim = 0; dim < NDIM; dim++) {
		X[dim] = fixed<int>(pos[dim]).raw();
	}
	if (multis.size()) {
		for (int i = 0; i < LP; i++) {
			Lacc[i] = 0.f;
		}
		const auto cnt1 = multis.size();
		int n;
		for (int j = 0; j < cnt1; j += simd_float::size()) {
			n = 0;
			for (int k = 0; k < simd_float::size(); k++) {
				if (j + k < cnt1) {
					const multipole mpole = multis[j + k].get_multi();
					const auto other_pos = multis[j + k].get_pos();
					for (int dim = 0; dim < NDIM; dim++) {
						Y[dim][k] = fixed<int>(other_pos[dim]).raw();
					}
					for (int i = 0; i < MP; i++) {
						M[i][k] = mpole[i];
					}
					n++;
				} else {
					const auto other_pos = multis[cnt1 - 1].get_pos();
					for (int dim = 0; dim < NDIM; dim++) {
						Y[dim][k] = fixed<int>(other_pos[dim]).raw();
					}
					for (int i = 0; i < MP; i++) {
						M[i][k] = 0.f;
					}
				}
			}
			interacts += n;
			for (int dim = 0; dim < NDIM; dim++) {
				dX[dim] = distance(X[dim], Y[dim]);
			}
			flops += n * 6;
			flops += n * green_ewald(D, dX);
			flops += n * multipole_interaction(Lacc, M, D, params.full_eval);
		}
		for (int k = 0; k < simd_float::size(); k++) {
			for (int i = 0; i < LP; i++) {
				L[i] += Lacc[i][k];
			}
		}
	}
	if (params.full_eval) {
		kick_return_update_interactions_cpu(KR_EWCC, interacts, flops);
	}
}
