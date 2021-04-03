#include <cosmictiger/hpx.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/simd.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/kick_return.hpp>
#include <cosmictiger/sort.hpp>
#include <cosmictiger/tree_database.hpp>

#include <set>

#include <cmath>

particle_set* tree::particles;

#define CPU_LOAD (0)

static unified_allocator kick_params_alloc;
//pranges tree::covered_ranges;
static std::atomic<size_t> parts_covered;

timer tmp_tm;

void tree::set_particle_set(particle_set *parts) {
	particles = parts;
}

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

hpx::future<sort_return> tree::create_child(sort_params &params, bool try_thread) {
	tree_ptr id;
	id.dindex = tree_data_allocate();
	params.tptr = id;
	const auto nparts = params.parts.second - params.parts.first;
	bool thread = false;
	const size_t min_parts2thread = particles->size() / hpx::thread::hardware_concurrency() / OVERSUBSCRIPTION;
	thread = try_thread && (nparts >= min_parts2thread);
#ifdef TEST_STACK
	thread = false;
#endif
	if (!thread) {
		sort_return rc = tree::sort(params);
		rc.check = id;
		return hpx::make_ready_future(std::move(rc));
	} else {
		return hpx::async([id, params]() {
			auto rc = tree::sort(params);
			rc.check = id;
			return rc;
		});
	}
}

sort_return tree::sort(sort_params params) {
	const auto &opts = global().opts;
	static std::atomic<int> gpu_searches(0);
	size_t active_parts = 0;
	size_t active_nodes = 0;
	pair<size_t, size_t> parts;
	tree_ptr self = params.tptr;
	if (params.iamroot()) {
		gpu_searches = 0;
		int dummy;
		params.set_root();
		params.min_depth = ewald_min_level(params.theta, global().opts.hsoft);
//		printf("min ewald = %i\n", params.min_depth);
	}
	parts = params.parts;
	self.set_parts(parts);
	if (params.depth == TREE_MAX_DEPTH) {
		printf("Exceeded maximum tree depth\n");
		abort();
	}

	//  multi = params.allocs->multi_alloc.allocate();
	const auto& box = params.box;
#ifdef TEST_TREE
	bool failed = false;
	for (size_t i = parts.first; i < parts.second; i++) {
		particle p = particles->part(i);
		if (!box.contains(p.x)) {
			printf("Particle out of range !\n");
			printf("Box\n");
			for (int dim = 0; dim < NDIM; dim++) {
				printf("%e %e |", box.begin[dim], box.end[dim]);
			}
			printf("\n");
			printf("Particle\n");
			for (int dim = 0; dim < NDIM; dim++) {
				printf("%e ", p.x[dim].to_float());
			}
			printf("\n");
			//       abort();
			failed = true;
		}
	}
	if (failed) {
		abort();
	}
#endif
#ifdef TEST_STACK
	{
		uint8_t dummy;
		printf("Stack usaged = %li Depth = %li \n", &dummy - params.stack_ptr, params.depth);
	}
#endif
	sort_return rc;
	array<tree_ptr, NCHILD> children;
	if (parts.second - parts.first > global().opts.bucket_size
			|| (params.depth < params.min_depth && parts.second - parts.first > 0)) {
		std::array<fast_future<sort_return>, NCHILD> futs;
		{
			const auto size = parts.second - parts.first;
			auto child_params = params.get_children();
			const int xdim = params.depth % NDIM;
			auto part_handle = particles->get_virtual_particle_set();
			double xmid = (box.begin[xdim] + box.end[xdim]) / 2.0;
			size_t pmid;
			if (params.depth == 0) {
				particles->prepare_sort();
			}
			pmid = particles->sort_range(parts.first, parts.second, xmid, xdim);
			child_params[LEFT].box.end[xdim] = child_params[RIGHT].box.begin[xdim] = xmid;
			child_params[LEFT].parts.first = parts.first;
			child_params[LEFT].parts.second = child_params[RIGHT].parts.first = pmid;
			child_params[RIGHT].parts.second = parts.second;

			for (int ci = 0; ci < NCHILD; ci++) {
				futs[ci] = create_child(child_params[ci], ci == LEFT);
			}
		}
		std::array<multipole, NCHILD> Mc;
		std::array<array<fixed32, NDIM>, NCHILD> Xc;
		std::array<float, NCHILD> Rc;
		multipole M;
		rc.active_parts = 0;
		rc.active_nodes = 1;
		rc.stats.nparts = 0;
		rc.stats.max_depth = 0;
		rc.stats.min_depth = TREE_MAX_DEPTH;
		rc.stats.nnodes = 1;
		rc.stats.nleaves = 0;
		for (int ci = 0; ci < NCHILD; ci++) {
			sort_return this_rc = futs[ci].get();
			children[ci] = this_rc.check;
			Mc[ci] = this_rc.check.get_multi();
			Xc[ci] = this_rc.check.get_pos();
			rc.active_parts += this_rc.active_parts;
			rc.active_nodes += this_rc.active_nodes;
			rc.stats.nparts += this_rc.stats.nparts;
			rc.stats.nleaves += this_rc.stats.nleaves;
			rc.stats.nnodes += this_rc.stats.nnodes;
			rc.stats.max_depth = std::max(rc.stats.max_depth, this_rc.stats.max_depth);
			rc.stats.min_depth = std::min(rc.stats.min_depth, this_rc.stats.min_depth);
		}
		if (rc.active_nodes == 1) {
			rc.active_nodes = 0;
		}
		std::array<double, NDIM> com = { 0, 0, 0 };
		const auto &MR = Mc[RIGHT];
		const auto &ML = Mc[LEFT];
		M() = ML() + MR();
		double rleft = 0.0;
		double rright = 0.0;
		array<fixed32, NDIM> pos;
		for (int dim = 0; dim < NDIM; dim++) {
			com[dim] = (ML() * Xc[LEFT][dim].to_double() + MR() * Xc[RIGHT][dim].to_double()) / (ML() + MR());
			pos[dim] = com[dim];
			rleft += sqr(Xc[LEFT][dim].to_double() - com[dim]);
			rright += sqr(Xc[RIGHT][dim].to_double() - com[dim]);
		}
		std::array<double, NDIM> xl, xr;
		for (int dim = 0; dim < NDIM; dim++) {
			xl[dim] = Xc[LEFT][dim].to_double() - com[dim];
			xr[dim] = Xc[RIGHT][dim].to_double() - com[dim];
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
		self.set_leaf(false);
		self.set_children(children);
	} else {
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
		M = 0.0;
		float radius = 0.0;
		for (auto i = parts.first; i < parts.second; i++) {
			double this_radius = 0.0;
			M() += 1.0;
			for (int n = 0; n < NDIM; n++) {
				const auto xn = particles->pos(n, i).to_double() - com[n];
				this_radius += xn * xn;
				for (int m = n; m < NDIM; m++) {
					const auto xm = particles->pos(m, i).to_double() - com[m];
					const auto xnm = xn * xm;
					M(n, m) += xnm;
					for (int l = m; l > NDIM; l++) {
						const auto xl = particles->pos(l, i).to_double() - com[l];
						M(n, m, l) -= xnm * xl;
					}
				}
			}
			this_radius = std::sqrt(this_radius);
			radius = std::max(radius, (float) (this_radius));
		}
		rc.active_parts = 0;
		rc.active_nodes = 0;
		for (size_t k = parts.first; k < parts.second; k++) {
			if (particles->rung(k) >= params.min_rung) {
				rc.active_parts++;
				rc.active_nodes = 1;
			}
		}
		rc.stats.nparts = parts.second - parts.first;
		rc.stats.max_depth = rc.stats.min_depth = params.depth;
		rc.stats.nnodes = 1;
		rc.stats.nleaves = 1;
		for (int ci = 0; ci < NCHILD; ci++) {
			children[ci].dindex = -1;
		}
		self.set_pos(pos);
		self.set_radius(radius);
		self.set_multi(M);
		self.set_leaf(true);
		self.set_children(children);
	}
	active_parts = rc.active_parts;
	active_nodes = rc.active_nodes;
	self.set_active_parts(active_parts);
	self.set_active_nodes(active_nodes);
	rc.stats.e_depth = params.min_depth;
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
	const auto parts = get_parts();
	const auto part_begin = parts.first;
	const auto part_end = parts.second;
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
			new (new_params) kick_params_type;
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

//int num_kicks = 0;
hpx::future<void> tree::kick(kick_params_type * params_ptr) {
	kick_params_type &params = *params_ptr;
	tree_ptr self = params.tptr;
	const size_t active_parts = self.get_active_parts();
	if (!active_parts && !params.full_eval) {
		return hpx::make_ready_future();
	}
	auto& F = params.F;
	auto& phi = params.Phi;
	const auto parts = self.get_parts();
	if (params.depth == 0) {
		kick_timer.start();
//		covered_ranges.clear();
		parts_covered = 0;
		tmp_tm.start();
		size_t dummy, total_mem;
		CUDA_CHECK(cudaMemGetInfo(&dummy, &total_mem));
		total_mem /= 8;
		size_t used_mem = (sizeof(vel_type) + sizeof(fixed32) * NDIM) * particles->size();
		double oversubscription = std::max(2.0, (double) used_mem / total_mem);
		const int block_count = oversubscription * global().cuda_kick_occupancy
				* global().cuda.devices[0].multiProcessorCount + 0.5;
//	/	printf( "Seeking %i blocks\n", block_count);
		size_t active_nodes = self.get_active_nodes();
		params.block_cutoff = std::max(active_nodes / block_count, (size_t) 1);
		if (active_parts < MIN_GPU_PARTS) {
			params.block_cutoff = 0;
			//		printf( "CPU ONLY\n");
		}
		tree_database_set_readonly();
		particles->prepare_kick();
	}
	if (self.is_leaf()) {
		for (int i = 0; i < parts.second - parts.first; i++) {
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
				const auto other_parts = checks[ci].get_parts();
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
		const bool try_thread = parts.second - parts.first > TREE_MIN_PARTS2THREAD;
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
			const auto other_parts = children[RIGHT].get_parts();
//			covered_ranges.add_range(parts);
			parts_covered += other_parts.second - other_parts.first;
			if (parts_covered == global().opts.nparts) {
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
			const auto other_parts = children[LEFT].get_parts();
//			covered_ranges.add_range(parts);
			parts_covered += other_parts.second - other_parts.first;
			if (parts_covered == global().opts.nparts) {
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
		for (int k = 0; k < parts.second - parts.first; k++) {
			const auto this_rung = particles->rung(k + parts.first);
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
				//		printf( "%li\n", params.stack_top - (uintptr_t)(&dummy));
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
					particles->force(dim, k + parts.first) = F[dim][k];
				}
				particles->pot(k + parts.first) = phi[k];
#endif
				if (this_rung >= params.rung) {
					float dt = params.t0 / (1 << this_rung);
					if (!params.first) {
						particles->vel(k + parts.first).p.x += 0.5 * dt * F[0][k];
						particles->vel(k + parts.first).p.y += 0.5 * dt * F[1][k];
						particles->vel(k + parts.first).p.z += 0.5 * dt * F[2][k];
					}
					float fmag = 0.0;
					for (int dim = 0; dim < NDIM; dim++) {
						fmag += sqr(F[dim][k]);
					}
					fmag = sqrtf(fmag);
					assert(fmag > 0.0);
					dt = std::min(params.eta * std::sqrt(params.scale * params.hsoft / fmag), params.t0);
					int new_rung = std::max(
							std::max(std::max(int(std::ceil(std::log(params.t0 / dt) * invlog2)), this_rung - 1), params.rung),
							MIN_RUNG);
					dt = params.t0 / (1 << new_rung);
					particles->vel(k + parts.first).p.x += 0.5 * dt * F[0][k];
					particles->vel(k + parts.first).p.y += 0.5 * dt * F[1][k];
					particles->vel(k + parts.first).p.z += 0.5 * dt * F[2][k];
					max_rung = std::max(max_rung, new_rung);
					particles->set_rung(new_rung, k + parts.first);
				}
				if (params.full_eval) {
					kick_return_update_pot_cpu(phi[k], F[0][k], F[1][k], F[2][k]);
				}
			}
			kick_return_update_rung_cpu(particles->rung(k + parts.first));

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
	int max_oc = global().cuda_kick_occupancy * global().cuda.devices[0].multiProcessorCount;
	max_blocks_active = 2 * max_oc;

	do {
		timer.stop();
		if (timer.read() > 0.05) {
			printf("%c %i in queue: %i active: %i complete\r", waiting_char[counter % 4], gpu_queue.size(), blocks_active,
					blocks_completed);
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
				deleters->push_back(tmp.params->tmp.to_device(stream));
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
			cuda_execute_kick_kernel(gpu_params, kicks.size(), stream);
			completions.push_back(std::function<bool()>([=]() {
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
					//		printf("Done executing\n");
					return true;
				} else {
					return false;
				}
			}));
			kick_grid_size = std::min((int) gpu_queue.size(), kick_grid_size);
		}
		int i = 0;
		while (i < completions.size()) {
			if (completions[i]()) {
				completions[i] = completions.back();
				completions.pop_back();
			} else {
				i++;
			}
		}
	} while (completions.size() || gpu_queue.size());
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
	parts_covered += gpu.parts.second - gpu.parts.first;
	if (parts_covered == global().opts.nparts) {
		gpu_daemon();
	}
	return std::move(fut);

}
