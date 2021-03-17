#include <cosmictiger/hpx.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/simd.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/kick_return.hpp>
#include <cosmictiger/sort.hpp>

#include <cmath>

particle_set* tree::particles;

#define CPU_LOAD (0)

static unified_allocator kick_params_alloc;

timer tmp_tm;

#define NWAVE 1
#define GPUOS 4

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

hpx::future<sort_return> tree::create_child(sort_params &params) {
	static std::atomic<int> threads_used(hpx_rank() == 0 ? 1 : 0);
	tree_ptr id;
	//  id.rank = 0;
	id.ptr = (uintptr_t) params.allocs->tree_alloc.allocate();
	CHECK_POINTER(id.ptr);
	const auto nparts = params.parts.second - params.parts.first;
	bool thread = false;
	if (nparts > TREE_MIN_PARTS2THREAD) {
		if (++threads_used <= OVERSUBSCRIPTION * hpx::thread::hardware_concurrency()) {
			thread = true;
		} else {
			threads_used--;
		}
	}
//	if( params.depth <= cpu_sort_depth()) {
//		thread = true;
//	}
#ifdef TEST_STACK
	thread = false;
#endif
	if (!thread) {
		sort_return rc = ((tree*) (id.ptr))->sort(params);
		rc.check = id;
		return hpx::make_ready_future(std::move(rc));
	} else {
		params.allocs = std::make_shared<tree_alloc>();
		return hpx::async([id, params]() {
			auto rc = ((tree*) (id.ptr))->sort(params);
			rc.check = id;
			threads_used--;
			return rc;
		});
	}
}

sort_return tree::sort(sort_params params) {
	const auto &opts = global().opts;
	static std::atomic<int> gpu_searches(0);
	active_parts = 0;
	active_nodes = 0;
	if (params.iamroot()) {
		gpu_searches = 0;
		int dummy;
		params.set_root();
		params.min_depth = ewald_min_level(global().opts.theta, global().opts.hsoft);
//		printf("min ewald = %i\n", params.min_depth);
	}
	{
//		const auto bnds = params.get_bounds();
///		parts.first = bnds.first;
//		parts.second = bnds.second;
		parts = params.parts;
	}
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
	if (parts.second - parts.first > MAX_BUCKET_SIZE
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

			/*	if (params.key_end - params.key_begin == 1) {
			 const auto &box = params.box;
			 int radix_depth = (int(log(double(size + 1) / MAX_BUCKET_SIZE) / log(2) + TREE_RADIX_CUSHION));
			 radix_depth = std::min(std::max(radix_depth, TREE_RADIX_MIN), TREE_RADIX_MAX) + params.depth;
			 const auto radix_begin = morton_key(box.begin, radix_depth);
			 std::array<fixed64, NDIM> tmp;
			 for (int dim = 0; dim < NDIM; dim++) {
			 tmp[dim] = box.end[dim] - fixed32::min();
			 }
			 const auto radix_end = morton_key(tmp, radix_depth) + 1;
			 auto bounds = particles->local_sort(parts.first, parts.second, radix_depth, radix_begin, radix_end);
			 assert(bounds[0] >= parts.first);
			 assert(bounds[bounds.size() - 1] <= parts.second);
			 auto bndptr = std::make_shared<decltype(bounds)>(std::move(bounds));
			 for (int ci = 0; ci < NCHILD; ci++) {
			 child_params[ci].bounds = bndptr;
			 }
			 child_params[LEFT].key_begin = 0;
			 child_params[LEFT].key_end = child_params[RIGHT].key_begin = (radix_end - radix_begin) / 2;
			 child_params[RIGHT].key_end = (radix_end - radix_begin);
			 }*/
			for (int ci = 0; ci < NCHILD; ci++) {
				futs[ci] = create_child(child_params[ci]);
			}
		}
		std::array<multipole, NCHILD> Mc;
		std::array<fixed32*, NCHILD> Xc;
		std::array<float, NCHILD> Rc;
		auto &M = (multi);
		rc.active_parts = 0;
		rc.active_nodes = 1;
		for (int ci = 0; ci < NCHILD; ci++) {
			sort_return this_rc = futs[ci].get();
			children[ci] = this_rc.check;
			Mc[ci] = ((tree*) this_rc.check)->multi;
			Xc[ci] = ((tree*) this_rc.check)->pos.data();
			children[ci] = this_rc.check;
			rc.active_parts += this_rc.active_parts;
			rc.active_nodes += this_rc.active_nodes;
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
		rleft = std::sqrt(rleft) + ((tree*) children[LEFT])->radius;
		rright = std::sqrt(rright) + ((tree*) children[RIGHT])->radius;
		radius = std::max(rleft, rright);
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
	} else {
		std::array<double, NDIM> com = { 0, 0, 0 };
		for (auto i = parts.first; i < parts.second; i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				com[dim] += particles->pos(dim, i).to_double();
			}
		}
		for (int dim = 0; dim < NDIM; dim++) {
			com[dim] /= (parts.second - parts.first);
			pos[dim] = com[dim];

		}
		auto &M = (multi);
		M = 0.0;
		radius = 0.0;
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
	}
	active_parts = rc.active_parts;
	active_nodes = rc.active_nodes;
	return rc;
}

hpx::lcos::local::mutex tree::mtx;
hpx::lcos::local::mutex tree::gpu_mtx;

void tree::cleanup() {
	shutdown_daemon = true;
	while (daemon_running) {
		hpx::this_thread::yield();
	}
}

hpx::future<void> tree_ptr::kick(kick_params_type *params_ptr, bool thread) {
	kick_params_type &params = *params_ptr;
	const auto part_begin = ((tree*) (*this))->parts.first;
	const auto part_end = ((tree*) (*this))->parts.second;
	const auto num_active = ((tree*) (*this))->active_nodes;
	const auto sm_count = global().cuda.devices[0].multiProcessorCount;
	const auto gpu_partcnt = global().opts.nparts / (sm_count * KICK_OCCUPANCY);
	bool use_cpu_block = false;
	if (num_active <= params.block_cutoff && num_active) {
		return ((tree*) ptr)->send_kick_to_gpu(params_ptr);
	} else if (thread) {
		kick_params_type *new_params;
		new_params = (kick_params_type*) kick_params_alloc.allocate(sizeof(kick_params_type));
		new (new_params) kick_params_type;
		*new_params = *params_ptr;
		auto func = [this, new_params]() {
			int dummy;
			auto rc = ((tree*) ptr)->kick(new_params);
			new_params->kick_params_type::~kick_params_type();
			kick_params_alloc.deallocate(new_params);
			return rc;
		};
		auto fut = hpx::async(std::move(func));
		return std::move(fut);
	} else {
		//	const int max_threads = OVERSUBSCRIPTION * hpx::threads::hardware_concurrency();
		//	static std::atomic<int> used_threads(0);
		//	if (used_threads++ > max_threads) {
		//		used_threads--;
		//		thread = false;
		//	}
		if (thread) {
			kick_params_type *new_params;
			new_params = (kick_params_type*) kick_params_alloc.allocate(sizeof(kick_params_type));
			new (new_params) kick_params_type;
			*new_params = *params_ptr;
			auto func = [this, new_params]() {
				auto rc = ((tree*) ptr)->kick(new_params);
				//		used_threads--;
					new_params->kick_params_type::~kick_params_type();
					kick_params_alloc.deallocate(new_params);
					return rc;
				};
			auto fut = hpx::async(std::move(func));
			return std::move(fut);
		} else {
			auto fut = hpx::make_ready_future(((tree*) ptr)->kick(params_ptr));
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

//int num_kicks = 0;
hpx::future<void> tree::kick(kick_params_type * params_ptr) {
	kick_params_type &params = *params_ptr;
	if (!active_nodes && !params.full_eval) {
		return hpx::make_ready_future();
	}
	auto& F = params.F;
	auto& phi = params.Phi;
	if (params.depth == 0) {
		kick_timer.start();
		tmp_tm.start();
		const auto sm_count = global().cuda.devices[0].multiProcessorCount;
		//	auto cpu_load = NWAVE * CPU_LOAD;
		const int target_max = GPUOS * NWAVE * sm_count * KICK_OCCUPANCY;	// + cpu_load;
		int pcnt = active_nodes;
		int count;
		do {
			count = compute_block_count(pcnt, true, params.full_eval);
			//	printf("%i\n", count);
			if (count < target_max) {
				pcnt = pcnt / 2;
			}
		} while (count < target_max && pcnt);
		params.block_cutoff = std::max(pcnt * 2, 16);
		kick_block_count = compute_block_count(params.block_cutoff, true, params.full_eval);
		//	printf("%i %i\n", (int) kick_block_count, params.block_cutoff);
		managed_allocator<tree>::set_device(0);
	}
	if (children[0].ptr == 0) {
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
	for (int dim = 0; dim < NDIM; dim++) {
		const auto x1 = pos[dim];
		const auto x2 = Lpos[dim];
		dx[dim] = distance(x1, x2);
	}
	L <<= dx;

	const auto theta2 = sqr(params.theta);
	array<tree_ptr*, N_INTERACTION_TYPES> all_checks;
	auto &multis = params.multi_interactions;
	auto &parti = params.part_interactions;
	auto &next_checks = params.next_checks;
	int ninteractions = is_leaf() ? 4 : 2;
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
			for (int ci = 0; ci < checks.size(); ci++) {
				const auto other_radius = checks[ci].get_radius();
				const auto other_parts = ((tree*) checks[ci])->parts;
				const auto other_pos = checks[ci].get_pos();
				float d2 = 0.f;
				for (int dim = 0; dim < NDIM; dim++) {
					d2 += sqr(distance(other_pos[dim], pos[dim]));
				}
				if (ewald_dist) {
					d2 = std::max(d2, (float) EWALD_MIN_DIST2);
				}
				const auto myradius = SINK_BIAS * (radius + params.hsoft);
				const auto R1 = sqr(other_radius + myradius + params.hsoft);                 // 2
				const auto R2 = sqr(other_radius * params.theta + myradius + params.hsoft);
				const auto R3 = sqr(other_radius + (myradius * params.theta / SINK_BIAS) + params.hsoft);
				const bool far1 = R1 < theta2 * d2;
				const bool far2 = R2 < theta2 * d2;
				const bool far3 = R3 < theta2 * d2;
				const bool isleaf = checks[ci].is_leaf();
				if (far1 || (direct && far3 && (other_parts.second - other_parts.first >= PC_MIN_PARTS))) {
					multis.push_back(checks[ci]);
				} else if ((far2 || direct) && isleaf) {
					parti.push_back(checks[ci]);
				} else if (isleaf) {
					next_checks.push_back(checks[ci]);
				} else {
					const auto child_checks = checks[ci].get_children().get();
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
	if (!is_leaf()) {
// printf("4\n");
		const bool try_thread = parts.second - parts.first > TREE_MIN_PARTS2THREAD;
		int depth0 = params.depth;
		params.depth++;
		params.dchecks.push_top();
		params.echecks.push_top();
		params.L[params.depth] = L;
		params.Lpos[params.depth] = pos;
		array<hpx::future<void>, NCHILD> futs;
		futs[LEFT] = children[LEFT].kick(params_ptr, try_thread);

//  printf("5\n");
		params.dchecks.pop_top();
		params.echecks.pop_top();
		params.L[params.depth] = L;
		futs[RIGHT] = children[RIGHT].kick(params_ptr, false);
		params.depth--;
		if (params.depth != depth0) {
			printf("error\n");
			abort();
		}
		return hpx::when_all(futs.begin(), futs.end()).then([](hpx::future<std::vector<hpx::future<void>>> futfut) {
			auto futs = futfut.get();
			futs[LEFT].get();
			futs[RIGHT].get();
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
				shift_expansion(L, g, this_phi, dx);
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
						for (int dim = 0; dim < NDIM; dim++) {
							particles->vel(dim, k + parts.first) += 0.5 * dt * F[dim][k];
						}
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
							0);
					dt = params.t0 / (1 << new_rung);
					for (int dim = 0; dim < NDIM; dim++) {
						particles->vel(dim, k + parts.first) += 0.5 * dt * F[dim][k];
					}
					max_rung = std::max(max_rung, new_rung);
					particles->set_rung(new_rung, k + parts.first);
				}
				kick_return_update_pot_cpu(phi[k], F[0][k], F[1][k], F[2][k]);
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
	static bool skip;
	static bool ewald_skip;
	static double wait_time = 0.0e-3;
	static int min_ewald;
	if (first_call) {
		//	printf("Starting gpu daemon\n");
		timer.reset();
		timer.start();
		first_call = false;
		min_ewald = 1;
	}
	timer.stop();
	if (timer.read() > wait_time) {
		timer.reset();
		timer.start();
		if (gpu_queue.size() == kick_block_count) {
			kick_timer.stop();
			//	printf("Time to GPU = %e\n", kick_timer.read());
			using promise_type = std::vector<std::shared_ptr<hpx::lcos::local::promise<void>>>;
			std::shared_ptr<promise_type> gpu_promises;
			//		std::shared_ptr<promise_type> cpu_promises[NWAVE];
			gpu_promises = std::make_shared<promise_type>();
			auto deleters = std::make_shared<std::vector<std::function<void()>> >();
			unified_allocator calloc;
			kick_params_type* gpu_params;
			std::vector<gpu_kick> kicks;
			while (gpu_queue.size()) {
				kicks.push_back(gpu_queue.pop());
			}
			gpu_params = (kick_params_type*) calloc.allocate(kicks.size() * sizeof(kick_params_type));
			auto stream = get_stream();
			std::sort(kicks.begin(), kicks.end(), [](const gpu_kick& a, const gpu_kick& b) {
				return a.parts.first < b.parts.first;
			});
			particles->prepare_kick(kicks[0].parts.first, kicks.back().parts.second, stream);
			for (int i = 0; i < kicks.size(); i++) {
				auto tmp = std::move(kicks[i]);
				deleters->push_back(tmp.params->dchecks.to_device(stream));
				deleters->push_back(tmp.params->echecks.to_device(stream));
				memcpy(gpu_params + i, tmp.params, sizeof(kick_params_type));
				auto tmpparams = tmp.params;
				deleters->push_back([tmpparams]() {
					kick_params_alloc.deallocate(tmpparams);
				});
				gpu_promises->push_back(std::move(tmp.promise));
			}
			deleters->push_back([calloc, gpu_params]() {
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
					for (int i = 0; i < kicks.size(); i++) {
						(*gpu_promises)[i]->set_value();
					}
					//		printf("Done executing\n");
					return true;
				} else {
					return false;
				}
			}));
		} else {
			min_ewald = std::max(min_ewald / 2, 1);
		}
	} else {
		timer.start();
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
	if (!shutdown_daemon) {
		hpx::async(gpu_daemon);
	} else {
		first_call = true;
		daemon_running = false;
	}
}

hpx::future<void> tree::send_kick_to_gpu(kick_params_type * params) {
	if (!daemon_running) {
		std::lock_guard<hpx::lcos::local::mutex> lock(gpu_mtx);
		if (!daemon_running) {
			shutdown_daemon = false;
			daemon_running = true;
			hpx::async([]() {
				gpu_daemon();
			});
		}
	}

	gpu_kick gpu;
	tree_ptr me;
	me.ptr = (uintptr_t) (this);
//   me.rank = hpx_rank();
	kick_params_type *new_params;
	new_params = (kick_params_type*) kick_params_alloc.allocate(sizeof(kick_params_type));
	new (new_params) kick_params_type();
	*new_params = *params;
	new_params->tptr = me;
	gpu.params = new_params;
	auto fut = gpu.promise->get_future();
	gpu.parts = parts;
	gpu_queue.push(std::move(gpu));

	return std::move(fut);

}

int tree::compute_block_count(size_t cutoff, bool root, bool full_eval) {
	if (!active_nodes && !full_eval) {
		return 0;
	}
	if (active_nodes <= cutoff && !root) {
		return 1;
	} else if (!is_leaf()) {
		auto left = ((tree*) children[LEFT])->compute_block_count(cutoff, false,full_eval);
		auto right = ((tree*) children[RIGHT])->compute_block_count(cutoff, false,full_eval);
		return left + right;
	} else {
		return 0;
	}
}

hpx::future<int32_t> tree::send_ewald_to_gpu(kick_params_type * params) {
	if (!daemon_running) {
		std::lock_guard<hpx::lcos::local::mutex> lock(gpu_mtx);
		if (!daemon_running) {
			shutdown_daemon = false;
			daemon_running = true;
			hpx::async([]() {
				gpu_daemon();
			});
		}
	}
	gpu_ewald gpu;
	tree_ptr me;
	me.ptr = (uintptr_t) (this);
//  me.rank = hpx_rank();
	params->tptr = me;
	gpu.params = params;
	auto fut = gpu.promise.get_future();
	gpu_ewald_queue.push(std::move(gpu));

	return std::move(fut);

}

