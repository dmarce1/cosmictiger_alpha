#include <cosmictiger/hpx.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/simd.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/kick_return.hpp>
#include <cosmictiger/groups.hpp>
#include <cosmictiger/tree_database.hpp>

#include <set>

#include <cmath>

HPX_PLAIN_ACTION(tree::sort,tree_sort_action);

#define CPU_LOAD (0)

static unified_allocator kick_params_alloc;
//pranges tree::covered_ranges;
static std::atomic<size_t> parts_covered;

timer tmp_tm;

static std::atomic<int> kick_block_count;

CUDA_EXPORT inline int ewald_min_level(double theta, double h) {
	int lev = 12;
	while (1) {
		int N = 1 << (lev / NDIM);
		double dx = EWALD_MIN_DIST * N;
		double a;
		constexpr double ffac = 1.01;
		if (lev % NDIM == 0) {
			a = std::sqrt(3) + ffac * h;
		} else if (lev % NDIM == 1) {
			a = 1.5 + ffac * h;
		} else {
			a = std::sqrt(1.5) + ffac * h;
		}
		double r = (1.0 + SINK_BIAS) * a / theta + h * N;
		if (dx > r) {
			break;
		}
		lev++;
	}
	return lev;
}

fast_future<sort_return> tree::create_child(sort_params &params, bool try_thread) {
	fast_future<sort_return> fut;
	tree_ptr id;
	particle_server pserv;
	auto& particles = pserv.get_particle_sets();
	size_t nparts = 0;
	for (int pi = 0; pi < NPART_TYPES; pi++) {
		nparts += params.parts[pi].second - params.parts[pi].first;
	}
	int target_rank = pserv.index_to_rank(params.parts[0].first);
	if (target_rank != hpx_rank()) {
		fut = hpx::async<tree_sort_action>(hpx_localities()[target_rank], params);
	} else {
		bool thread = false;
		const size_t min_parts2thread = particles.size() / hpx::thread::hardware_concurrency() / OVERSUBSCRIPTION;
		thread = try_thread && (nparts >= min_parts2thread);
		if (!thread) {
			sort_return rc = tree::sort(params);
			fut.set_value(std::move(rc));
		} else {
			fut = hpx::async([id, params]() {
				sort_params new_params = params;
				new_params.alloc = std::make_shared<tree_allocator>();
				auto rc = tree::sort(new_params);
				return rc;
			});
		}
	}
	return std::move(fut);
}

sort_return tree::sort(sort_params params) {
	const auto &opts = global().opts;
	size_t active_parts = 0;
	size_t active_nodes = 0;
	parts_type parts;
	if (params.iamroot()) {
		int dummy;
		params.set_root();
		params.min_depth = ewald_min_level(params.theta, global().opts.hsoft);
		params.alloc = std::make_shared<tree_allocator>();
	}
	tree_ptr self = params.alloc->allocate();
	particle_server pserv;
	auto& particles = pserv.get_particle_sets();
	parts = params.parts;
	self.set_parts(parts);
	if (params.depth == TREE_MAX_DEPTH) {
		ERROR()
		;
	}

	const auto& box = params.box;
	sort_return rc;
	array<tree_ptr, NCHILD> children;
	const int max_part = params.group_sort ? GROUP_BUCKET_SIZE : global().opts.bucket_size;
	size_t total_parts = parts.size();
	const bool sph = global().opts.sph;
	const bool groups = params.group_sort;
	bool all_local = true;
	if (total_parts > max_part || (params.depth < params.min_depth && total_parts > 0)) {
		std::array<fast_future<sort_return>, NCHILD> futs;
		{
			auto child_params = params.get_children();
			const int xdim = params.depth % NDIM;
			double xmid = (box.begin[xdim] + box.end[xdim]) / 2.0;
			child_params[LEFT].box.end[xdim] = child_params[RIGHT].box.begin[xdim] = xmid;
			for (int pi = 0; pi < NPART_TYPES; pi++) {
				const size_t pmid = pserv.sort(pi, parts[pi].first, parts[pi].second, xmid, xdim);
				child_params[LEFT].parts[pi].first = parts[pi].first;
				child_params[LEFT].parts[pi].second = child_params[RIGHT].parts[pi].first = pmid;
				child_params[RIGHT].parts[pi].second = parts[pi].second;
			}
			for (int ci = 0; ci < NCHILD; ci++) {
				futs[ci] = create_child(child_params[ci], ci == LEFT);
			}
		}

		std::array<multipole, NCHILD> Mc;
		std::array<array<fixed32, NDIM>, NCHILD> Xc;
		std::array<float, NCHILD> Rc;
		std::array<range, NCHILD> sph_ranges;
		std::array<range, NCHILD> ranges;
		multipole M;
		if (!params.group_sort) {
			rc.active_nodes = 1;
			rc.active_parts = 0;
			rc.stats.nparts = 0;
			rc.stats.max_depth = 0;
			rc.stats.min_depth = TREE_MAX_DEPTH;
			rc.stats.nnodes = 1;
			rc.stats.nleaves = 0;
		} else {
			rc.active_nodes = 0;
		}
		for (int ci = 0; ci < NCHILD; ci++) {
			sort_return this_rc = futs[ci].get();
			children[ci] = this_rc.check;
			all_local = all_local && (hpx_rank() == this_rc.check.rank) && this_rc.all_local;
			rc.active_nodes += this_rc.active_nodes;
			if (sph || groups) {
				ranges[ci] = this_rc.check.get_range();
			}
			if (!params.group_sort) {
				Mc[ci] = this_rc.check.get_multi();
				Xc[ci] = this_rc.check.get_pos();
				if (sph) {
					sph_ranges[ci] = this_rc.check.get_sph_range();
				}
				rc.active_parts += this_rc.active_parts;
				rc.stats.nparts += this_rc.stats.nparts;
				rc.stats.nleaves += this_rc.stats.nleaves;
				rc.stats.nnodes += this_rc.stats.nnodes;
				rc.stats.max_depth = std::max(rc.stats.max_depth, this_rc.stats.max_depth);
				rc.stats.min_depth = std::min(rc.stats.min_depth, this_rc.stats.min_depth);
			}
		}
		if (rc.active_nodes == 1) {
			rc.active_nodes = 0;
		}
		if (!params.group_sort) {
			std::array<double, NDIM> com = { 0, 0, 0 };
			const auto &MR = Mc[RIGHT];
			const auto &ML = Mc[LEFT];
			M() = ML() + MR();
			double rleft = 0.0;
			double rright = 0.0;
			array<fixed32, NDIM> pos;
			std::array<double, NDIM> xl, xr;
			for (int dim = 0; dim < NDIM; dim++) {
				xl[dim] = Xc[LEFT][dim].to_double();
				xr[dim] = Xc[RIGHT][dim].to_double();
			}
			for (int dim = 0; dim < NDIM; dim++) {
				auto& Xl = xl[dim];
				auto& Xr = xr[dim];
				com[dim] = (ML() * Xl + MR() * Xr) / (ML() + MR());
				Xl -= com[dim];
				Xr -= com[dim];
				rleft += sqr(Xl);
				rright += sqr(Xr);
				pos[dim] = com[dim];
			}
			M = (ML >> xl) + (MR >> xr);
			rleft = std::sqrt(rleft) + children[LEFT].get_radius();
			rright = std::sqrt(rright) + children[RIGHT].get_radius();
			float radius = std::max(rleft, rright);
			float rmax = 0.0;
			const auto corners = params.box.get_corners();
			for (int ci = 0; ci < NCORNERS; ci++) {
				double d = 0.0;
				for (int dim = 0; dim < NDIM; dim++) {
					d += sqr(com[dim] - corners[ci][dim].to_double());
				}
				rmax = std::max((float) std::sqrt(d), rmax);
			}
			radius = std::min(radius, rmax);

			//    printf("x      = %e\n", pos[0].to_float());
			//   printf("y      = %e\n", pos[1].to_float());
			//  printf("z      = %e\n", pos[2].to_float());
			// printf("radius = %e\n", radius);
			self.set_pos(pos);
			self.set_radius(radius);
			self.set_multi(M);
		}
		self.set_leaf(false);
		self.set_children(children);
		if (sph) {
			range myrange;
			for (int dim = 0; dim < NDIM; dim++) {
				myrange.begin[dim] = std::min(sph_ranges[LEFT].begin[dim], sph_ranges[RIGHT].begin[dim]);
				myrange.end[dim] = std::max(sph_ranges[LEFT].end[dim], sph_ranges[RIGHT].end[dim]);
			}
			self.set_sph_range(myrange);
		}
		if (groups || sph) {
			range myrange;
			for (int dim = 0; dim < NDIM; dim++) {
				myrange.begin[dim] = std::min(ranges[LEFT].begin[dim], ranges[RIGHT].begin[dim]);
				myrange.end[dim] = std::max(ranges[LEFT].end[dim], ranges[RIGHT].end[dim]);
			}
			self.set_range(myrange);
		}
	} else {
		if (!params.group_sort) {
			std::array<double, NDIM> com = { 0, 0, 0 };
			array<fixed32, NDIM> pos;
			double total_weight = 0;
			for (int pi = 0; pi < NPART_TYPES; pi++) {
				const double wt = particles.weights[pi];
				for (auto i = parts[pi].first; i < parts[pi].second; i++) {
					total_weight += wt;
					for (int dim = 0; dim < NDIM; dim++) {
						com[dim] += wt * particles.sets[pi]->pos(dim, i).to_double();
					}
				}
			}
			if (total_weight) {
				const auto twinv = 1.0 / total_weight;
				for (int dim = 0; dim < NDIM; dim++) {
					com[dim] *= twinv;
					pos[dim] = com[dim];

				}
			} else {
				for (int dim = 0; dim < NDIM; dim++) {
					com[dim] = (box.begin[dim] + box.end[dim]) * 0.5;
				}
			}
			multipole M;
			M = 0.0;
			float radius = 0.0;
			for (int pi = 0; pi < NPART_TYPES; pi++) {
				const float wt = particles.weights[pi];
				for (auto i = parts[pi].first; i < parts[pi].second; i++) {
					double this_radius = 0.0;
					M() += wt;
					for (int n = 0; n < NDIM; n++) {
						const auto xn = particles.sets[pi]->pos(n, i).to_double() - com[n];
						this_radius += xn * xn;
						for (int m = n; m < NDIM; m++) {
							const auto xm = particles.sets[pi]->pos(m, i).to_double() - com[m];
							const auto xnm = xn * xm;
							M(n, m) += wt * xnm;
							for (int l = m; l > NDIM; l++) {
								const auto xl = particles.sets[pi]->pos(l, i).to_double() - com[l];
								M(n, m, l) -= wt * xnm * xl;
							}
						}
					}
					this_radius = std::sqrt(this_radius);
					radius = std::max(radius, (float) (this_radius));
				}
			}
			rc.active_parts = 0;
			rc.active_nodes = 0;
			for (int pi = 0; pi < NPART_TYPES; pi++) {
				for (size_t k = parts[pi].first; k < parts[pi].second; k++) {
					if (particles.sets[pi]->rung(k) >= params.min_rung) {
						rc.active_parts++;
						rc.active_nodes = 1;
					}
				}
			}
			rc.stats.nparts = total_parts;
			rc.stats.max_depth = rc.stats.min_depth = params.depth;
			rc.stats.nnodes = 1;
			rc.stats.nleaves = 1;
			self.set_pos(pos);
			self.set_radius(radius);
			self.set_multi(M);
		} else {
			rc.active_nodes = 1;
		}
		for (int ci = 0; ci < NCHILD; ci++) {
			children[ci].dindex = -1;
		}
		self.set_leaf(true);
		self.set_children(children);
		if (sph && !groups) {
			range myrange;
			for (int dim = 0; dim < NDIM; dim++) {
				myrange.begin[dim] = 1.0;
				myrange.end[dim] = 0.0;
			}
			const auto& these_particles = particles.baryon;
			for (size_t i = parts[BARY_SET].first; i != parts[BARY_SET].second; i++) {
				const float h = these_particles.smooth_len(i);
				for (int dim = 0; dim < NDIM; dim++) {
					const float pos = these_particles.pos(dim, i).to_float();
					myrange.begin[dim] = std::min(myrange.begin[dim], pos - h);
					myrange.end[dim] = std::max(myrange.end[dim], pos + h);
				}
			}
			self.set_sph_range(myrange);
		}
		if (groups || sph) {
			range myrange;
			for (int dim = 0; dim < NDIM; dim++) {
				myrange.begin[dim] = 1.0;
				myrange.end[dim] = 0.0;
			}
			for (int pi = 0; pi < NPART_TYPES; pi++) {
				if (!groups && pi == CDM_SET) {
					continue;
				}
				const auto& these_particles = particles.sets[pi];
				for (size_t i = parts[pi].first; i != parts[pi].second; i++) {
					for (int dim = 0; dim < NDIM; dim++) {
						const float pos = these_particles->pos(dim, i).to_float();
						myrange.begin[dim] = std::min(myrange.begin[dim], pos);
						myrange.end[dim] = std::max(myrange.end[dim], pos);
					}
				}
			}
			self.set_range(myrange);
		}

	}
	active_nodes = rc.active_nodes;
	self.set_active_nodes(active_nodes);
	if (!params.group_sort) {
		active_parts = rc.active_parts;
		self.set_active_parts(active_parts);
		self.set_active_nodes(active_nodes);
		rc.stats.e_depth = params.min_depth;
	}
	rc.check = self;
	const int npart_types = global().opts.sph ? 2 : 1;
	for (int pi = 0; pi < npart_types; pi++) {
		all_local = all_local && (pserv.index_to_rank(parts[pi].first) != hpx_rank());
		all_local = all_local && (pserv.index_to_rank(parts[pi].second) != hpx_rank());
	}
	self.set_all_local(all_local);
	rc.all_local = all_local;
	return rc;
}

hpx::lcos::local::mutex tree::mtx;
hpx::lcos::local::mutex tree::gpu_mtx;

void tree::cleanup() {
	tree_data_clear();
	shutdown_daemon = true;
	while (daemon_running) {
		hpx::this_thread::yield();
	}
}

hpx::future<void> tree_ptr::kick(kick_params_type *params_ptr, bool thread) {
	static const bool use_cuda = global().opts.cuda;
	kick_params_type &params = *params_ptr;
	const auto num_active = get_active_nodes();
	const auto sm_count = global().cuda.devices[0].multiProcessorCount;
	if (use_cuda && num_active <= params.block_cutoff && num_active) {
		return tree::send_kick_to_gpu(params_ptr);
	} else {
		static std::atomic<int> used_threads(0);
		if (thread) {
			const int max_threads = OVERSUBSCRIPTION * hpx::threads::hardware_concurrency();
			if (used_threads++ > max_threads) {
				used_threads--;
				thread = false;
			}
		}
		if (thread) {
			kick_params_type *new_params;
			new_params = (kick_params_type*) kick_params_alloc.allocate(sizeof(kick_params_type));
			new (new_params) kick_params_type();
			*new_params = *params_ptr;
			tree_ptr me = *this;
			auto func = [me,new_params]() {
				auto rc = tree::kick(new_params);
				used_threads--;
				new_params->kick_params_type::~kick_params_type();
				kick_params_alloc.deallocate(new_params);
				return rc;
			};
			auto fut = hpx::async(std::move(func));
			return std::move(fut);
		} else {
			auto fut = hpx::make_ready_future(tree::kick(params_ptr));
			return fut;
		}
	}
}

#define CC_CP_DIRECT 0
#define CC_CP_EWALD 1
#define PC_PP_DIRECT 2
#define PC_PP_EWALD 3
#define N_INTERACTION_TYPES 4

timer kick_timer;

#define MIN_WORK 0

hpx::future<void> tree::kick(kick_params_type * params_ptr) {
	kick_params_type &params = *params_ptr;
	tree_ptr self = params.tptr;
	const size_t active_parts = self.get_active_parts();
	if (!active_parts && !params.full_eval) {
		return hpx::make_ready_future();
	}
	auto& F = params.F;
	auto& phi = params.Phi;
	parts_type parts;
	parts = self.get_parts();
//	printf( "%i %i\n", parts[1].first, parts[1].second);
//	abort();
	if (params.depth == 0) {
		kick_timer.start();
		parts_covered = 0;
		tmp_tm.start();
		size_t dummy, total_mem;
		CUDA_CHECK(cudaMemGetInfo(&dummy, &total_mem));
		total_mem /= 8;
		size_t used_mem = (sizeof(rung_t) + NDIM * sizeof(float) + sizeof(fixed32) * NDIM) * params.part_sets->size()
				+ tree_data_bytes_used();
		double oversubscription = std::max(2.0, (double) used_mem / total_mem);
		int numBlocks;
		CUDA_CHECK(
				cudaOccupancyMaxActiveBlocksPerMultiprocessor ( &numBlocks, cuda_kick_kernel, KICK_BLOCK_SIZE, sizeof(cuda_kick_shmem) ));
		const int block_count = oversubscription * numBlocks * global().cuda.devices[0].multiProcessorCount + 0.5;
		size_t active_nodes = self.get_active_nodes();
		params.block_cutoff = std::max(active_nodes / block_count, (size_t) 1);
		int min_gpu_nodes = block_count / 8;
		if (active_nodes < min_gpu_nodes) {
			params.block_cutoff = 0;
		}
		tree_database_set_readonly();
	}
	if (self.is_leaf()) {
		for (int i = 0; i < parts.size(); i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				F[dim][i] = 0.f;
			}
			phi[i] = -PHI0;
		}
	}

#ifdef TEST_CHECKLIST_TIME
	static timer tm;
#endif
	//  printf( "%li\n", params.depth);
	assert(params.depth < TREE_MAX_DEPTH);
	auto &L = params.L[params.depth];
	const auto &Lpos = params.Lpos[params.depth];
	array<float, NDIM> dx;
	const auto pos = self.get_pos();
	for (int dim = 0; dim < NDIM; dim++) {
		const auto x1 = pos[dim];
		const auto x2 = Lpos[dim];
		dx[dim] = distance(x1, x2);
	}
	shift_expansion(L, dx, params.full_eval);

	const auto theta2 = sqr(params.theta);
	array<tree_ptr*, N_INTERACTION_TYPES> all_checks;
	auto &multis = params.multi_interactions;
	auto &parti = params.part_interactions;
	auto &next_checks = params.next_checks;
	int ninteractions = params.tptr.is_leaf() ? 4 : 2;
	bool found_ewald = false;
	for (int type = 0; type < ninteractions; type++) {
		const bool ewald_dist = type == PC_PP_EWALD || type == CC_CP_EWALD;
		auto &checks = ewald_dist ? params.echecks : params.dchecks;
		const bool direct = type == PC_PP_EWALD || type == PC_PP_DIRECT;
		parti.resize(0);
		multis.resize(0);
		do {
			next_checks.resize(0);
#ifdef TEST_CHECKLIST_TIME
			tm.start();
#endif
			const auto th = params.theta * std::max(params.hsoft, MIN_DX);
			for (int ci = 0; ci < checks.size(); ci++) {
				const auto other_radius = checks[ci].get_radius();
				const auto other_pos = checks[ci].get_pos();
				float d2 = 0.f;
				for (int dim = 0; dim < NDIM; dim++) {
					d2 += sqr(distance(other_pos[dim], pos[dim]));
				}
				if (ewald_dist) {
					d2 = std::max(d2, (float) EWALD_MIN_DIST2);
				}
				const float myradius = SINK_BIAS * self.get_radius();
				const auto R1 = sqr(std::max(other_radius + myradius + th, MIN_DX));                 // 2
				const auto R2 = sqr(std::max(other_radius * params.theta + myradius + th, MIN_DX));
				const auto R3 = sqr(std::max(other_radius + (myradius * params.theta / SINK_BIAS) + th, MIN_DX));
				const bool far1 = R1 < theta2 * d2;
				const bool far2 = R2 < theta2 * d2;
				const bool far3 = R3 < theta2 * d2;
				const bool isleaf = checks[ci].is_leaf();
				if (far1 || (direct && far3)) {
					multis.push_back(checks[ci]);
				} else if ((far2 || direct) && isleaf) {
					parti.push_back(checks[ci]);
				} else if (isleaf) {
					next_checks.push_back(checks[ci]);
				} else {
					const auto child_checks = checks[ci].get_children();
					next_checks.push_back(child_checks[LEFT]);
					next_checks.push_back(child_checks[RIGHT]);
				}
			}
#ifdef TEST_CHECKLIST_TIME
			tm.stop();
#endif
			checks.resize(0);
			for (int i = 0; i < next_checks.size(); i++) {
				checks.push(next_checks[i]);
			}
		} while (direct && checks.size());
		clock_t tm;
		switch (type) {
		case CC_CP_DIRECT:
			cpu_cc_direct(params_ptr);
			cpu_cp_direct(params_ptr);
			break;
		case CC_CP_EWALD:
			cpu_cc_ewald(params_ptr);
			break;
		case PC_PP_DIRECT:
			cpu_pc_direct(params_ptr);
			cpu_pp_direct(params_ptr);
			break;
		case PC_PP_EWALD:
			break;
		}

	}
//  printf( "%li\n", params.depth);
	if (!params.tptr.is_leaf()) {
// printf("4\n");
		const bool try_thread = parts.size() > TREE_MIN_PARTS2THREAD;
		array<hpx::future<void>, NCHILD> futs;
		futs[LEFT] = hpx::make_ready_future();
		futs[RIGHT] = hpx::make_ready_future();
		auto children = self.get_children();
		if ((children[LEFT].get_active_parts() && children[RIGHT].get_active_parts()) || params.full_eval) {
			int depth0 = params.depth;
			params.depth++;
			params.dchecks.push_top();
			params.echecks.push_top();
			params.L[params.depth] = L;
			params.Lpos[params.depth] = pos;
			params.tptr = children[LEFT];
			futs[LEFT] = children[LEFT].kick(params_ptr, try_thread);
			params.dchecks.pop_top();
			params.echecks.pop_top();
			params.L[params.depth] = L;
			params.tptr = children[RIGHT];
			futs[RIGHT] = children[RIGHT].kick(params_ptr, false);
			params.depth--;
		} else if (children[LEFT].get_active_parts()) {
			parts_type other_parts;
			other_parts = children[RIGHT].get_parts();
//			covered_ranges.add_range(parts);
			parts_covered += other_parts.size();
			if (parts_covered == params.part_sets->size()) {
				gpu_daemon();
			}
			int depth0 = params.depth;
			params.depth++;
			params.L[params.depth] = L;
			params.Lpos[params.depth] = pos;
			params.tptr = children[LEFT];
			futs[LEFT] = children[LEFT].kick(params_ptr, false);
			params.depth--;
		} else if (children[RIGHT].get_active_parts()) {
			parts_type other_parts;
			other_parts = children[LEFT].get_parts();
//			covered_ranges.add_range(parts);
			parts_covered += other_parts.size();
			if (parts_covered == params.part_sets->size()) {
				gpu_daemon();
			}
			int depth0 = params.depth;
			params.depth++;
			params.L[params.depth] = L;
			params.Lpos[params.depth] = pos;
			params.tptr = children[RIGHT];
			futs[RIGHT] = children[RIGHT].kick(params_ptr, false);
			params.depth--;
		}
		int depth = params.depth;
		return hpx::when_all(futs.begin(), futs.end()).then([depth](hpx::future<std::vector<hpx::future<void>>> futfut) {
			auto futs = futfut.get();
			futs[LEFT].get();
			futs[RIGHT].get();
			if( depth == 0 ) {
				tree_database_unset_readonly();
			}
		});
	} else {
		int max_rung = 0;
		const auto invlog2 = 1.0f / logf(2);
		for (int pi = 0; pi < NPART_TYPES; pi++) {
			auto& particles = params.part_sets->sets[pi];
			const auto offset = parts.index_offset(pi);
			for (int k = 0; k < parts[pi].second - parts[pi].first; k++) {
				const size_t index = k + parts[pi].first;
				const auto this_rung = particles->rung(index);
				if (this_rung >= params.rung || params.full_eval) {
					array<float, NDIM> g;
					float this_phi;
					for (int dim = 0; dim < NDIM; dim++) {
						const auto x2 = pos[dim];
						const auto x1 = particles->pos(dim, k + parts[pi].first);
						dx[dim] = distance(x1, x2);
					}
					shift_expansion(L, g, this_phi, dx, params.full_eval);
					//		int dummy;
					//		printf( "%li\n", params.stack_top - (uintptr_t)(&dummy));
					for (int dim = 0; dim < NDIM; dim++) {
						F[dim][k + offset] += g[dim];
					}
					phi[k] += this_phi;
					for (int dim = 0; dim < NDIM; dim++) {
						F[dim][k + offset] *= params.G * params.M;
					}
					phi[k + offset] *= params.G * params.M;
					if (params.groups) {
						cpu_groups_kick_update(particles->group(index), phi[k]);
					}
#ifdef TEST_FORCE
					for (int dim = 0; dim < NDIM; dim++) {
						particles->force(dim, k + parts[pi].first) = F[dim][k];
					}
					particles->pot(k + parts[pi].first) = phi[k];
#endif
					if (this_rung >= params.rung) {
						float dt = params.t0 / (1 << this_rung);
						if (!params.first) {
							particles->vel(0, index) += 0.5 * dt * F[0][k];
							particles->vel(1, index) += 0.5 * dt * F[1][k];
							particles->vel(2, index) += 0.5 * dt * F[2][k];
						}
						float fmag = 0.0;
						for (int dim = 0; dim < NDIM; dim++) {
							fmag += sqr(F[dim][k]);
						}
						fmag = sqrtf(fmag);
						assert(fmag > 0.0);
						dt = std::min(params.eta * std::sqrt(params.scale * params.hsoft / fmag), params.t0);
						int new_rung = std::max(
								std::max(std::max(int(std::ceil(std::log(params.t0 / dt) * invlog2)), this_rung - 1),
										params.rung),
								MIN_RUNG);
						dt = params.t0 / (1 << new_rung);
						particles->vel(0, index) += 0.5 * dt * F[0][k];
						particles->vel(1, index) += 0.5 * dt * F[1][k];
						particles->vel(2, index) += 0.5 * dt * F[2][k];
						max_rung = std::max(max_rung, new_rung);
						particles->set_rung(new_rung, index);
					}
					if (params.full_eval) {
						kick_return_update_pot_cpu(phi[k], F[0][k], F[1][k], F[2][k]);
					}
				}
				kick_return_update_rung_cpu(particles->rung(index));
			}
		}
		//	rc.rung = max_rung;
		return hpx::make_ready_future();
	}
}

lockfree_queue<gpu_kick, GPU_QUEUE_SIZE> tree::gpu_queue;
lockfree_queue<gpu_ewald, GPU_QUEUE_SIZE> tree::gpu_ewald_queue;
std::atomic<bool> tree::daemon_running(false);
std::atomic<bool> tree::shutdown_daemon(false);

void cuda_execute_kick_kernel(kick_params_type **params, int grid_size);

void tree::gpu_daemon() {
	static bool first_call = true;
	static std::vector<std::function<bool()>> completions;
	static timer timer;
	static int counter = 0;
	static char waiting_char[4] = { '-', '\\', '|', '/' };
	static int blocks_completed;
	static int blocks_active;
	static int max_blocks_active;
	blocks_active = blocks_completed = 0;
	timer.reset();
	timer.start();
	first_call = false;
	int numBlocks;
	CUDA_CHECK(
			cudaOccupancyMaxActiveBlocksPerMultiprocessor ( &numBlocks, cuda_kick_kernel, KICK_BLOCK_SIZE, sizeof(cuda_kick_shmem) ));
	int max_oc = numBlocks * global().cuda.devices[0].multiProcessorCount;
	max_blocks_active = 2 * max_oc;
	bool first_pass = true;
	bool alldone;
	do {
		timer.stop();
		if (timer.read() > 0.05) {
//			printf("%c %i in queue: %i active: %i complete                                  \r", waiting_char[counter % 4],
//					(int) gpu_queue.size(), blocks_active, blocks_completed);
			counter++;
			timer.reset();
		}
		timer.start();
		int kick_grid_size = std::min((int) gpu_queue.size(), max_oc);
		while (gpu_queue.size() && kick_grid_size && blocks_active + kick_grid_size <= max_blocks_active) {
			using promise_type = std::vector<std::shared_ptr<hpx::lcos::local::promise<void>>>;
			std::shared_ptr<promise_type> gpu_promises;
			gpu_promises = std::make_shared<promise_type>();
			auto deleters = std::make_shared<std::vector<std::function<void()>> >();
			unified_allocator calloc;
			kick_params_type* gpu_params;
			std::vector<gpu_kick> kicks;
			while (gpu_queue.size()) {
				kicks.push_back(gpu_queue.pop());
			}
			std::sort(kicks.begin(), kicks.end(), [&](const gpu_kick& a, const gpu_kick& b) {
				return a.parts[CDM_SET].first < b.parts[CDM_SET].first;
			});
			while (kicks.size() > kick_grid_size) {
				gpu_queue.push(kicks.back());
				kicks.pop_back();
			}
			blocks_active += kicks.size();
			gpu_params = (kick_params_type*) calloc.allocate(kicks.size() * sizeof(kick_params_type));
			auto stream = get_stream();
			for (int i = 0; i < kicks.size(); i++) {
				auto tmp = std::move(kicks[i]);
				deleters->push_back(tmp.params->dchecks.to_device(stream));
				deleters->push_back(tmp.params->echecks.to_device(stream));
				deleters->push_back(tmp.params->multi_interactions.to_device(stream));
				deleters->push_back(tmp.params->part_interactions.to_device(stream));
				deleters->push_back(tmp.params->next_checks.to_device(stream));
				deleters->push_back(tmp.params->opened_checks.to_device(stream));
				memcpy(gpu_params + i, tmp.params, sizeof(kick_params_type));
				auto tmpparams = tmp.params;
				deleters->push_back([tmpparams]() {
					kick_params_alloc.deallocate(tmpparams);
				});
				gpu_promises->push_back(std::move(tmp.promise));
			}
			const auto sz = kicks.size();
			deleters->push_back([calloc, gpu_params, sz]() {
				unified_allocator calloc;
				calloc.deallocate(gpu_params);
			});
			auto completed = std::make_shared<bool>(false);
			if (first_pass) {
				first_pass = false;
				kick_constants consts;
				consts.G = gpu_params[0].G;
				consts.M = gpu_params[0].M;
				consts.eta = gpu_params[0].eta;
				consts.full_eval = gpu_params[0].full_eval;
				consts.h = gpu_params[0].hsoft;
				consts.rung = gpu_params[0].rung;
				consts.scale = gpu_params[0].scale;
				consts.t0 = gpu_params[0].t0;
				consts.theta = gpu_params[0].theta;
				consts.first = gpu_params[0].first;
				consts.groups = gpu_params[0].groups;
				cuda_set_kick_constants(consts, *gpu_params[0].part_sets);
			}
			cuda_execute_kick_kernel(gpu_params, kicks.size(), stream);
			completions.push_back(std::function<bool()>([=]() {
				if( !((*completed))) {
					if( cudaStreamQuery(stream) == cudaSuccess) {
						CUDA_CHECK(cudaStreamSynchronize(stream));
						cleanup_stream(stream);
						for (auto &d : *deleters) {
							d();
						}
						blocks_completed += sz;
						blocks_active -= sz;
						for (int i = 0; i < sz; i++) {
							(*gpu_promises)[i]->set_value();
						}
						*completed = true;
					}
				}
				return *completed;
			}));
			kick_grid_size = std::min((int) gpu_queue.size(), kick_grid_size);
		}
		int i = 0;
		alldone = true;
		for (int i = 0; i < completions.size(); i++) {
			if (!completions[i]()) {
				alldone = false;
				break;
			}
		}
	} while (!alldone || gpu_queue.size());
}

hpx::future<void> tree::send_kick_to_gpu(kick_params_type * params) {
	/*if (!daemon_running) {
	 std::lock_guard<hpx::lcos::local::mutex> lock(gpu_mtx);
	 if (!daemon_running) {
	 shutdown_daemon = false;
	 daemon_running = true;
	 hpx::async([]() {
	 gpu_daemon();
	 });
	 }
	 }*/

	gpu_kick gpu;
	kick_params_type *new_params;
	new_params = (kick_params_type*) kick_params_alloc.allocate(sizeof(kick_params_type));
	new (new_params) kick_params_type();
	*new_params = *params;
	gpu.params = new_params;
	auto fut = gpu.promise->get_future();
	gpu.parts = params->tptr.get_parts();
	gpu_queue.push(std::move(gpu));
//	covered_ranges.add_range(parts);
	parts_covered += gpu.parts.size();
	if (parts_covered == params->part_sets->size()) {
		gpu_daemon();
	}
	return std::move(fut);

}
