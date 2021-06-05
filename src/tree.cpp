#include <cosmictiger/hpx.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/simd.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/kick_return.hpp>
#include <cosmictiger/groups.hpp>
#include <cosmictiger/particle_server.hpp>
#include <cosmictiger/tree_database.hpp>
#include <cosmictiger/spherical_harmonic.hpp>

#include <set>

#include <cmath>

HPX_PLAIN_ACTION(tree::sort, tree_sort_action);
HPX_PLAIN_ACTION(tree::kick_remote, tree_kick_remote_action);

static unified_allocator kick_params_alloc;
//pranges tree::covered_ranges;
static std::atomic<part_int> parts_covered;
static range my_domain;
static bool dry_run;
static std::unordered_set<tree_ptr, tree_hash> remote_parts;

int cuda_kick_occupancy() {
	int num_blocks;
	CUDA_CHECK(
			cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, (const void*) cuda_kick_kernel, KICK_BLOCK_SIZE, sizeof(cuda_kick_shmem)));
	return num_blocks;
}

void tree::add_parts_covered(part_iters num) {
	const part_int count = (parts_covered += num.second - num.first);
	static particle_server pserv;
	static const auto& parts = pserv.get_particle_set();
//	PRINT( "%i %i\n", (int) parts_covered, parts.size());
	if (count == parts.size()) {
		if (dry_run) {
//			PRINT("Doing global to local\n");
			pserv.global_to_local(std::move(remote_parts));
		} else {
//			PRINT("Sending tree pieces to gpu\n");
			gpu_daemon();
		}

	}
	if (parts_covered > parts.size()) {
		ERROR()
		;
	}
}

timer tmp_tm;

static std::atomic<int> kick_block_count;

static bool all_checks_local(const stack_vector<tree_ptr>& checks) {
	bool local = true;
	for (int i = 0; i < checks.size(); i++) {
		if (checks[i].rank != hpx_rank()) {
			local = false;
			break;
		}
	}
	if (local) {
		for (int i = 0; i < checks.size(); i++) {
			auto proc_range = checks[i].get_proc_range();
			if (proc_range.second - proc_range.first > 1) {
				local = false;
				break;
			}
		}
	}
	return local;
}

static bool all_checks_local_or_decomp(const stack_vector<tree_ptr>& checks) {
	bool local = true;
	for (int i = 0; i < checks.size(); i++) {
		auto proc_range = checks[i].get_proc_range();
		if (checks[i].rank != hpx_rank() && proc_range.second - proc_range.first == 1) {
			local = false;
			break;
		}
	}
	return local;
}

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
	particle_server pserv;
	auto* particles = &pserv.get_particle_set();
	const auto nparts = params.parts.second - params.parts.first;
	const auto decomp = params.procs.second - params.procs.first > 1;
	bool thread = false;
	const part_int min_parts2thread = particles->size() / hpx::thread::hardware_concurrency() / OVERSUBSCRIPTION;
	thread = try_thread && (nparts >= min_parts2thread || decomp);
#ifdef TEST_STACK
	thread = false;
#endif
	if (params.procs.first != hpx_rank()) {
		fut = hpx::async<tree_sort_action>(hpx_localities()[params.procs.first], params);
	} else {

		if (!thread) {
			sort_return rc = tree::sort(params);
			fut.set_value(std::move(rc));
		} else {
			fut = hpx::async([params]() {
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
	particle_server pserv;
	auto* particles = &pserv.get_particle_set();
	static std::atomic<int> gpu_searches(0);
	size_t active_parts = 0;
	size_t active_nodes = 0;
	part_iters parts;

	if (params.iamroot()) {
		gpu_searches = 0;
		int dummy;
		params.set_root();
		params.min_depth = ewald_min_level(params.theta, global().opts.hsoft);
		params.alloc = std::make_shared<tree_allocator>();

//		PRINT("min ewald = %i\n", params.min_depth);
	}
	tree_ptr self;
	self.dindex = params.alloc->allocate();
	self.rank = hpx_rank();
	if (params.local_root) {
//		PRINT("Sorting local root on %i with %i particles\n", hpx_rank(), particles->size());
		params.parts.first = 0;
		params.parts.second = particles->size();
		self.set_local_root(true);
		my_domain = pserv.get_domain_bounds().find_range(params.procs);
	} else {
		self.set_local_root(false);
	}
	parts = params.parts;
	self.set_parts(parts);
	self.set_proc_range(params.procs.first, params.procs.second);
	if (params.depth == TREE_MAX_DEPTH) {
		PRINT("Exceeded maximum tree depth\n");
		abort();
	}

	//  multi = params.allocs->multi_alloc.allocate();
	const auto& box = params.box;

	sort_return rc;
	array<tree_ptr, NCHILD> children;
	const bool domain_decomp = (params.procs.second - params.procs.first) > 1;
	int max_part = (params.group_sort ? GROUP_BUCKET_SIZE : global().opts.bucket_size);
//	PRINT("%li %li %i %i\n", params.procs.first, params.procs.second, params.parts.first, params.parts.second);
	if (domain_decomp || parts.second - parts.first > max_part
			|| (params.depth < params.min_depth && parts.second - parts.first > 0)) {
		std::array<fast_future<sort_return>, NCHILD> futs;
		auto child_params = params.get_children();
		if (!domain_decomp) {
			const auto size = parts.second - parts.first;
			double max_span = 0.0;
			int xdim;
			for (int dim = 0; dim < NDIM; dim++) {
				const auto span = box.end[dim] - box.begin[dim];
				if (span > max_span) {
					max_span = span;
					xdim = dim;
				}
			}
			auto part_handle = particles->get_virtual_particle_set();
			double xmid = (box.begin[xdim] + box.end[xdim]) / 2.0;
//			double xmid = particles->find_middle(parts.first, parts.second, xdim).to_double();
			part_int pmid;
			pmid = particles->sort_range(parts.first, parts.second, xmid, xdim);
			child_params[LEFT].box.end[xdim] = child_params[RIGHT].box.begin[xdim] = xmid;
			child_params[LEFT].parts.first = parts.first;
			child_params[LEFT].parts.second = child_params[RIGHT].parts.first = pmid;
			child_params[RIGHT].parts.second = parts.second;
		} else {
			const int pmid = (params.procs.first + params.procs.second) / 2;
			child_params[LEFT].procs.second = child_params[RIGHT].procs.first = pmid;
			for (int ci = 0; ci < NCHILD; ci++) {
				child_params[ci].box = pserv.get_domain_bounds().find_range(child_params[ci].procs);
				child_params[ci].local_root = (child_params[ci].procs.second - child_params[ci].procs.first == 1);
			}
		}
		for (int ci = 0; ci < NCHILD; ci++) {
			futs[ci] = create_child(child_params[ci], ci == LEFT);
		}

		std::array<multipole, NCHILD> Mc;
		std::array<array<fixed32, NDIM>, NCHILD> Xc;
		std::array<float, NCHILD> Rc;
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
			rc.active_nodes += this_rc.active_nodes;
			if (!params.group_sort) {
				Mc[ci] = this_rc.multi;
				Xc[ci] = this_rc.pos;
				Rc[ci] = this_rc.radius;
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
			auto MR = Mc[RIGHT];
			auto ML = Mc[LEFT];
			double ml = ML(0).real();
			double mr = MR(0).real();
		//	printf( "%e %e\n", mr, ml);
			double m = ml + mr;
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
				com[dim] = (ml * Xl + mr * Xr) / (ml + mr);
				Xl = -Xl + com[dim];
				Xr = -Xr + com[dim];
				rleft += sqr(Xl);
				rright += sqr(Xr);
				pos[dim] = com[dim];
			}
			translate_multipole<float, MORDER>(MR, MR, xr[0], xr[1], xr[2]);
			translate_multipole<float, MORDER>(ML, ML, xl[0], xl[1], xl[2]);
			auto M = ML + MR;
			rleft = std::sqrt(rleft) + Rc[LEFT];
			rright = std::sqrt(rright) + Rc[RIGHT];
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

			//    PRINT("x      = %e\n", pos[0].to_float());
			//   PRINT("y      = %e\n", pos[1].to_float());
			//  PRINT("z      = %e\n", pos[2].to_float());
			// PRINT("radius = %e\n", radius);
			self.set_pos(pos);
			self.set_radius(radius);
			self.set_multi(M);
		}
		self.set_leaf(false);
		self.set_children(children);
	} else {
		if (!params.group_sort) {
			std::array<double, NDIM> com = { 0, 0, 0 };
			array<fixed32, NDIM> pos;
			if (parts.second - parts.first != 0) {
				for (auto i = parts.first; i < parts.second; i++) {
					for (int dim = 0; dim < NDIM; dim++) {
						com[dim] += particles->pos(dim, i).to_double();
					}
				}
				for (int dim = 0; dim < NDIM; dim++) {
					com[dim] /= (parts.second - parts.first);
					pos[dim] = com[dim];

				}
			} else {
				for (int dim = 0; dim < NDIM; dim++) {
					com[dim] = (box.begin[dim] + box.end[dim]) * 0.5;
				}
			}
			multipole M;
			multipole Mtmp;
			M = 0.0;
			float radius = 0.0;
			sphericalY<float, 1> point;
			point(0).real() = 1.0f;
			point(0).imag() = 0.0f;
			for (auto i = parts.first; i < parts.second; i++) {
				const double x = com[0] - particles->pos(0, i).to_double();
				const double y = com[1] - particles->pos(1, i).to_double();
				const double z = com[2] - particles->pos(2, i).to_double();
				const double this_radius = sqrt(sqr(x - com[0]) + sqr(y - com[1]) + sqr(z - com[2]));
				translate_multipole<float,MORDER>(Mtmp, point, x, y, z);
				M = M + Mtmp;
				radius = std::max(radius, (float) (this_radius));
			}
//			printf( "%e\n", M(1,0).real());
			rc.active_parts = 0;
			rc.active_nodes = 0;
			for (part_int k = parts.first; k < parts.second; k++) {
				if (particles->rung(k) >= params.min_rung) {
					rc.active_parts++;
					rc.active_nodes = 1;
				}
			}
			rc.stats.nparts = parts.second - parts.first;
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
			children[ci].rank = -1;
			children[ci].dindex = -1;
		}
		self.set_leaf(true);
		self.set_children(children);
	}
	active_nodes = rc.active_nodes;
	self.set_active_nodes(active_nodes);
	if (!params.group_sort) {
		active_parts = rc.active_parts;
		self.set_active_parts(active_parts);
		self.set_active_nodes(active_nodes);
		rc.stats.e_depth = params.min_depth;
	} else {
		self.set_range(box);
	}
	rc.check = self;
	rc.multi = self.get_multi();
	rc.pos = self.get_pos();
	rc.radius = self.get_radius();
	return rc;
}

mutex_type tree::mtx;

void tree::cleanup() {
	tree_data_clear();
	shutdown_daemon = true;
	while (daemon_running) {
		hpx::this_thread::yield();
	}
}

void tree::kick_remote(kick_params_type params) {
	dry_run = params.dry_run;
	kick(&params);
}

hpx::future<void> tree_ptr::kick(kick_params_type *params_ptr, bool thread) {
	particle_server pserv;
	auto* particles = &pserv.get_particle_set();
	static const bool use_cuda = global().opts.cuda;
	kick_params_type &params = *params_ptr;
	const auto parts = get_parts();
	const auto proc_range = get_proc_range();
	const auto num_active = get_active_nodes();
	const auto sm_count = global().cuda.devices[0].multiProcessorCount;
	const bool isnt_global = proc_range.second - proc_range.first == 1;
	if (params.dry_run) {
		const bool all_local = all_checks_local(params.dchecks) && all_checks_local(params.echecks);
		if (all_local) {
			tree::add_parts_covered(parts);
			return hpx::make_ready_future();
		}
	}
#ifndef NDEBUG
	else {
		const bool is_local_root = local_root();
		if (proc_range.second - proc_range.first == 1 && !is_local_root) {
			const bool all_local = all_checks_local_or_decomp(params.dchecks)
					&& all_checks_local_or_decomp(params.echecks);
			assert(all_local);
		}
	}
#endif

	if (!params.dry_run && use_cuda && num_active <= params.block_cutoff && num_active && params.tptr.rank == hpx_rank()
			&& isnt_global) {
		return tree::send_kick_to_gpu(params_ptr);
	} else {
		if (params.tptr.rank != hpx_rank()) {
			return hpx::async<tree_kick_remote_action>(hpx_localities()[params.tptr.rank], params);
		} else {
			//	thread = thread
			//				&& ((proc_range.second - proc_range.first > 1) || (parts.second - parts.first > min_parts2thread));
			if (thread) {
				static std::atomic<int> counter(0);
//				printf( "Threading %i\n", (int) ++counter);
				kick_params_type *new_params;
				new_params = (kick_params_type*) kick_params_alloc.allocate(sizeof(kick_params_type));
				new (new_params) kick_params_type();
				*new_params = *params_ptr;
				auto func = [new_params]() {
					tree::kick(new_params);
					new_params->kick_params_type::~kick_params_type();
					kick_params_alloc.deallocate(new_params);
				};
				auto fut = hpx::async(std::move(func));
				return std::move(fut);
			} else {
				tree::kick(params_ptr);
			}
		}
	}
	return hpx::make_ready_future();
}

#define CC_CP_DIRECT 0
#define CC_CP_EWALD 1
#define PC_PP_DIRECT 2
#define PC_PP_EWALD 3
#define N_INTERACTION_TYPES 4

timer kick_timer;

#define MIN_WORK 0

void tree::kick(kick_params_type * params_ptr) {
	cuda_set_device();
	particle_server pserv;
	auto* particles = &pserv.get_particle_set();
	kick_params_type &params = *params_ptr;
	tree_ptr self = params.tptr;
	const size_t active_parts = self.get_active_parts();
	auto& F = params.F;
	auto& phi = params.Phi;
	const auto parts = self.get_parts();
	if (self.local_root()) {
//		PRINT("Starting local kick on %i\n", hpx_rank());
		dry_run = params.dry_run;
		if (!params.dry_run) {
			tree_data_global_to_local(params.dchecks);
			tree_data_global_to_local(params.echecks);
		}
		kick_timer.start();
		parts_covered = 0;
		tmp_tm.start();
		size_t dummy, total_mem;
		CUDA_CHECK(cudaMemGetInfo(&dummy, &total_mem));
		total_mem /= 8;
		size_t used_mem = (sizeof(rung_t) + NDIM * sizeof(float) + sizeof(fixed32) * NDIM) * particles->size()
				+ tree_data_bytes_used();
		double oversubscription = std::max(2.0, (double) used_mem / total_mem);
		int num_blocks = cuda_kick_occupancy();
		const int block_count = oversubscription * num_blocks * global().cuda.devices[0].multiProcessorCount + 0.5;
		size_t active_nodes = self.get_active_nodes();
		params.block_cutoff = std::max(active_nodes / block_count, (size_t) 1);
		int min_gpu_nodes = block_count / 8;
		if (active_nodes < min_gpu_nodes) {
			params.block_cutoff = 0;
		}
	}
	if (self.is_leaf()) {
		for (int i = 0; i < parts.second - parts.first; i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				F[dim][i] = 0.f;
			}
			phi[i] = -PHI0 / params.hsoft;
		}
	}

#ifdef TEST_CHECKLIST_TIME
	static timer tm;
#endif
	//  PRINT( "%li\n", params.depth);
	assert(params.depth < TREE_MAX_DEPTH);
	auto &L = params.L[params.depth];
	const auto &Lpos = params.Lpos[params.depth];
	array<float, NDIM> dx;
	const auto pos = self.get_pos();
	if (!params.dry_run) {
		for (int dim = 0; dim < NDIM; dim++) {
			const auto x1 = pos[dim];
			const auto x2 = Lpos[dim];
			dx[dim] = distance(x1, x2);
		}
		shift_expansion(L, dx, params.full_eval);
	}
	const auto theta2 = sqr(params.theta * (params.dry_run ? 1.0 : 1.0));

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
			const auto th = params.theta * params.hsoft;
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
				auto R1 = sqr(other_radius + myradius + th);                 // 2
				auto R2 = sqr(other_radius * params.theta + myradius + th);
				auto R3 = sqr(other_radius + (myradius * params.theta / SINK_BIAS) + th);
				if (params.dry_run) {
					R1 *= 1.001;
					R2 *= 0.999;
					R3 *= 1.001;
				}
				const bool far1 = R1 < theta2 * d2;
				const bool far2 = R2 < theta2 * d2;
				const bool far3 = R3 < theta2 * d2;
				const bool isleaf = checks[ci].is_leaf();
				if (far1 || (direct && far3 && !params.dry_run)) {
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
		if (params.dry_run) {
			std::lock_guard<mutex_type> lock(mtx);
			for (int i = 0; i < parti.size(); i++) {
				if (parti[i].rank != hpx_rank()) {
					remote_parts.insert(parti[i]);
				}
			}
			/*			for (int i = 0; i < multis.size(); i++) {
			 if (multis[i].rank != hpx_rank()) {
			 remote_parts.insert(multis[i]);
			 }
			 }*/
		} else {
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
	}
	if (!params.tptr.is_leaf()) {
		array<hpx::future<void>, NCHILD> futs;
		futs[LEFT] = hpx::make_ready_future();
		futs[RIGHT] = hpx::make_ready_future();
		auto children = self.get_children();
		const auto left_procs = children[LEFT].get_proc_range();
		const auto right_procs = children[RIGHT].get_proc_range();
		const bool left_branch = (left_procs.second - left_procs.first > 1) || children[LEFT].local_root()
				|| children[LEFT].get_active_parts();
		const bool right_branch = (right_procs.second - right_procs.first > 1) || children[RIGHT].local_root()
				|| children[RIGHT].get_active_parts();
		if ((left_branch && right_branch) || params.full_eval) {
			int depth0 = params.depth;
			params.depth++;
			params.dchecks.push_top();
			params.echecks.push_top();
			params.L[params.depth] = L;
			params.Lpos[params.depth] = pos;
			params.tptr = children[LEFT];
			futs[LEFT] = children[LEFT].kick(params_ptr, true);
			params.dchecks.pop_top();
			params.echecks.pop_top();
			params.L[params.depth] = L;
			params.tptr = children[RIGHT];
			futs[RIGHT] = children[RIGHT].kick(params_ptr, false);
			params.depth--;
		} else if (left_branch) {
			const auto other_parts = children[RIGHT].get_parts();
			add_parts_covered(other_parts);
			int depth0 = params.depth;
			params.depth++;
			params.L[params.depth] = L;
			params.Lpos[params.depth] = pos;
			params.tptr = children[LEFT];
			futs[LEFT] = children[LEFT].kick(params_ptr, false);
			params.depth--;
		} else if (right_branch) {
			const auto other_parts = children[LEFT].get_parts();
			add_parts_covered(other_parts);
			int depth0 = params.depth;
			params.depth++;
			params.L[params.depth] = L;
			params.Lpos[params.depth] = pos;
			params.tptr = children[RIGHT];
			futs[RIGHT] = children[RIGHT].kick(params_ptr, false);
			params.depth--;
		} else {
			add_parts_covered(children[RIGHT].get_parts());
			add_parts_covered(children[LEFT].get_parts());
		}
		int depth = params.depth;
		const bool dry = params.dry_run;
		futs[LEFT].get();
		futs[RIGHT].get();
		if (self.local_root()) {
//			PRINT("Freeing cache\n");
			if (!dry_run) {
				tree_data_free_cache();
				particles->resize_pos(particles->size());
			}
//			PRINT("%i\n", particles->size());
		}
		if (depth == 0) {
//			PRINT("Kick complete\n");
		}
	} else if (!params.dry_run) {
		int max_rung = 0;
		const auto invlog2 = 1.0f / logf(2);
		for (int k = 0; k < parts.second - parts.first; k++) {
			const part_int index = k + parts.first;
			const auto this_rung = particles->rung(index);
			if (this_rung >= params.rung || params.full_eval) {
				array<float, NDIM> g;
				float this_phi;
				for (int dim = 0; dim < NDIM; dim++) {
					const auto x2 = pos[dim];
					const auto x1 = particles->pos(dim, k + parts.first);
					dx[dim] = distance(x1, x2);
				}
				shift_expansion(L, g, this_phi, dx, params.full_eval);
				//		int dummy;
				//		PRINT( "%li\n", params.stack_top - (uintptr_t)(&dummy));
				for (int dim = 0; dim < NDIM; dim++) {
					F[dim][k] += g[dim];
				}
				phi[k] += this_phi;
				for (int dim = 0; dim < NDIM; dim++) {
					F[dim][k] *= params.G * params.M;
				}
				phi[k] *= params.G * params.M;
				if (params.groups) {
					cpu_groups_kick_update(particles->group(index), phi[k]);
				}
#ifdef TEST_FORCE
				for (int dim = 0; dim < NDIM; dim++) {
					particles->force(dim, k + parts.first) = F[dim][k];
				}
				particles->pot(k + parts.first) = phi[k];
#endif
				if (this_rung >= params.rung) {
					float dt = params.t0 / (1 << this_rung);
#ifndef CONFORMAL_TIME
					dt *= 1.f / params.scale;
#endif
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
#ifdef CONFORMAL_TIME
					dt = std::min(params.eta * std::sqrt(params.scale * params.hsoft / fmag), params.t0);
#else
					dt = std::min(params.eta * std::sqrt(params.scale * sqr(params.scale) * params.hsoft / fmag), params.t0);
#endif
					int new_rung = std::max(
							std::max(std::max(int(std::ceil(std::log(params.t0 / dt) * invlog2)), this_rung - 1), params.rung),
							MIN_RUNG);
					dt = params.t0 / (1 << new_rung);
#ifndef CONFORMAL_TIME
					dt *= 1.f / params.scale;
#endif
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
		add_parts_covered(parts);
	} else {
		add_parts_covered(parts);
	}
}

lockfree_queue<gpu_kick, GPU_QUEUE_SIZE> tree::gpu_queue;
lockfree_queue<gpu_ewald, GPU_QUEUE_SIZE> tree::gpu_ewald_queue;
std::atomic<bool> tree::daemon_running(false);
std::atomic<bool> tree::shutdown_daemon(false);

void cuda_execute_kick_kernel(kick_params_type **params, int grid_size);

void tree::gpu_daemon() {
	cuda_set_device();
	particle_server pserv;
	auto* particles = &pserv.get_particle_set();
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
	int num_blocks = cuda_kick_occupancy();
	int max_oc = num_blocks * global().cuda.devices[0].multiProcessorCount;
	max_blocks_active = 2 * max_oc;
	bool first_pass = true;
	bool alldone;
	do {
		timer.stop();
		if (timer.read() > 0.05) {
//			PRINT("%c %i in queue: %i active: %i complete                                  \r", waiting_char[counter % 4],
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
				return a.parts.first < b.parts.first;
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
				memcpy(consts.particles, particles, sizeof(particle_set));
				cuda_set_kick_constants(consts);
			}
			const int sindex = cuda_execute_kick_kernel(gpu_params, kicks.size(), stream);
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
						if( !check_kick_success(sindex) ) {
							PRINT( "Kick kernel failed to launch!\n");
							abort();
						}
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
	add_parts_covered(gpu.parts);
	return std::move(fut);

}
