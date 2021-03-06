#include <cosmictiger/hpx.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/simd.hpp>
#include <cosmictiger/gravity.hpp>

#include <cmath>

particle_set *tree::particles;

static unified_allocator kick_params_alloc;

timer tmp_tm;

void tree::set_particle_set(particle_set *parts) {
	particles = parts;
}

static int cuda_block_count;

CUDA_EXPORT inline int ewald_min_level(double theta, double h) {
	int lev = 12;
	while (1) {
		int N = 1 << (lev / NDIM);
		double dx = 0.25 * N;
		double a;
		if (lev % NDIM == 0) {
			a = std::sqrt(3);
		} else if (lev % NDIM == 1) {
			a = 1.5;
		} else {
			a = std::sqrt(1.5);
		}
		double r = (1.0 + SINK_BIAS) * a / theta + h * N;
		if (dx > r) {
			break;
		}
		lev++;
	}
	return lev;
}

int cuda_depth() {
	return 12;
}

hpx::future<sort_return> tree::create_child(sort_params &params) {
	static std::atomic<int> threads_used(hpx_rank() == 0 ? 1 : 0);
	tree_ptr id;
	//  id.rank = 0;
	id.ptr = (uintptr_t) params.allocs->tree_alloc.allocate();
	CHECK_POINTER(id.ptr);
	const auto nparts = (*params.bounds)[params.key_end] - (*params.bounds)[params.key_begin];
	bool thread = false;
	if (nparts > TREE_MIN_PARTS2THREAD) {
		if (++threads_used <= hpx::thread::hardware_concurrency()) {
			thread = true;
		} else {
			threads_used--;
		}
	}
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
	if (params.iamroot()) {
		int dummy;
		params.set_root();
		params.min_depth = ewald_min_level(global().opts.theta, global().opts.hsoft);
		printf("min ewald = %i\n", params.min_depth);
	}
	{
		const auto bnds = params.get_bounds();
		parts.first = bnds.first;
		parts.second = bnds.second;
	}
	if (params.depth == TREE_MAX_DEPTH) {
		printf("Exceeded maximum tree depth\n");
		abort();
	}

	//  multi = params.allocs->multi_alloc.allocate();
#ifdef TEST_TREE
	const auto &box = params.box;
	bool failed = false;
	for (size_t i = parts.first; i < parts.second; i++) {
		particle p = particles->part(i);
		if (!box.contains(p.x)) {
			printf("Particle out of range !\n");
			printf("Box\n");
			for (int dim = 0; dim < NDIM; dim++) {
				printf("%e %e |", box.begin[dim].to_float(), box.end[dim].to_float());
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
		// abort();
	}
#endif
#ifdef TEST_STACK
	{
		uint8_t dummy;
		printf("Stack usaged = %li Depth = %li \n", &dummy - params.stack_ptr, params.depth);
	}
#endif
	if (parts.second - parts.first > MAX_BUCKET_SIZE /*|| params.depth < params.min_depth*/) {
		std::array<fast_future<sort_return>, NCHILD> futs;
		{
			const auto size = parts.second - parts.first;
			auto child_params = params.get_children();
			if (params.key_end - params.key_begin == 1) {
#ifndef TEST_TREE
				const auto &box = params.box;
#endif
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
			}
			for (int ci = 0; ci < NCHILD; ci++) {
				futs[ci] = create_child(child_params[ci]);
			}
		}
		std::array<multipole, NCHILD> Mc;
		std::array<fixed32*, NCHILD> Xc;
		std::array<float, NCHILD> Rc;
		auto &M = (multi);
		for (int ci = 0; ci < NCHILD; ci++) {
			sort_return rc = futs[ci].get();
			children[ci] = rc.check;
			Mc[ci] = ((tree*) rc.check)->multi;
			Xc[ci] = ((tree*) rc.check)->pos.data();
			children[ci] = rc.check;
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
	}
	sort_return rc;
	return rc;
}

hpx::lcos::local::mutex tree::mtx;
hpx::lcos::local::mutex tree::gpu_mtx;

ewald_indices* tree::real_indices_ptr = nullptr;
ewald_indices* tree::four_indices_ptr = nullptr;
periodic_parts* tree::periodic_parts_ptr = nullptr;

void tree::cleanup() {
	shutdown_daemon = true;
	while (daemon_running) {
		hpx::this_thread::yield();
	}
}

hpx::future<kick_return> tree_ptr::kick(kick_params_type *params_ptr, bool thread, bool gpu) {
	kick_params_type &params = *params_ptr;
	const auto part_begin = ((tree*) (*this))->parts.first;
	const auto part_end = ((tree*) (*this))->parts.second;
	const auto sm_count = global().cuda.devices[0].multiProcessorCount;
	const auto gpu_partcnt = global().opts.nparts / (sm_count * KICK_OCCUPANCY);
	if (part_end - part_begin <= params.cuda_cutoff) {
//   if (params_ptr->depth == cuda_depth()) {
//   if (gpu) {
///		return ((tree*) ptr)->send_kick_to_gpu(params_ptr);
		thread = false;
	} //else {
	static std::atomic<int> threads_used(hpx_rank() == 0 ? 1 : 0);
	if (thread) {
		if (threads_used++ > 2 * hpx::thread::hardware_concurrency()) {
			//		thread = false;
			threads_used--;
		}
	}
	if (thread) {
		kick_params_type *new_params;
		new_params = (kick_params_type*) kick_params_alloc.allocate(sizeof(kick_params_type));
		new (new_params) kick_params_type;
		new_params->first = params_ptr->first;
		new_params->dchecks = params_ptr->dchecks.copy_top();
		new_params->echecks = params_ptr->echecks.copy_top();
		new_params->L[params_ptr->depth] = params_ptr->L[params_ptr->depth];
		new_params->Lpos[params_ptr->depth] = params_ptr->Lpos[params_ptr->depth];
		new_params->depth = params_ptr->depth;
		new_params->theta = params_ptr->theta;
		new_params->eta = params_ptr->eta;
		new_params->scale = params_ptr->scale;
		new_params->t0 = params_ptr->t0;
		new_params->rung = params_ptr->rung;
		new_params->cuda_cutoff = params_ptr->cuda_cutoff;
		auto func = [this, new_params]() {
			int dummy;
			new_params->stack_top = (uintptr_t) &dummy;
			auto rc = ((tree*) ptr)->kick(new_params);
			new_params->kick_params_type::~kick_params_type();
			kick_params_alloc.deallocate(new_params);
			//   threads_used--;
				return rc;
			};
		auto fut = hpx::async(std::move(func));
		return std::move(fut);
	} else {
		return hpx::make_ready_future(((tree*) ptr)->kick(params_ptr));
	}
//	}
}

#define CC_CP_DIRECT 0
#define CC_CP_EWALD 1
#define PC_PP_DIRECT 2
#define PC_PP_EWALD 3
#define N_INTERACTION_TYPES 4

//int num_kicks = 0;
hpx::future<kick_return> tree::kick(kick_params_type *params_ptr) {
	kick_params_type &params = *params_ptr;

	auto& F = params.F;
	if (params.depth == 0) {
		int dummy;
		params.stack_top = (uintptr_t) &dummy;
		tmp_tm.start();
		const auto sm_count = global().cuda.devices[0].multiProcessorCount;
		const int target_max = 2 * sm_count * KICK_OCCUPANCY;
		int pcnt = parts.second - parts.first;
		int count;
		do {
			count = compute_block_count(pcnt);
			if (count < target_max) {
				pcnt = pcnt / 2;
			}
		} while (count < target_max);
		params.cuda_cutoff = pcnt * 2;
		cuda_block_count = compute_block_count(params.cuda_cutoff);
	}
	if (children[0].ptr == 0) {
		for (int i = 0; i < parts.second - parts.first; i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				F[dim][i] = 0.f;
			}
		}
	}

#ifdef TEST_CHECKLIST_TIME
	static timer tm;
#endif
	kick_return rc;
	rc.flops = 0;
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
				const auto other_pos = checks[ci].get_pos();
				float d2 = 0.f;
				for (int dim = 0; dim < NDIM; dim++) {
					d2 += sqr(distance(other_pos[dim], pos[dim]));
				}
				if (ewald_dist) {
					d2 = std::max(d2, EWALD_MIN_DIST2);
				}
				const auto myradius = SINK_BIAS * (radius + params.hsoft);
				const auto R1 = sqr(other_radius + myradius + params.hsoft);                 // 2
				const auto R2 = sqr(other_radius * params.theta + myradius + params.hsoft);
				const auto R3 = sqr(other_radius + (myradius * params.theta / SINK_BIAS) + params.hsoft);
				const bool far1 = R1 < theta2 * d2;
				const bool far2 = R2 < theta2 * d2;
				const bool far3 = R3 < theta2 * d2;
				const bool isleaf = checks[ci].is_leaf();
				if (far1 || (direct) && far3) {
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

		switch (type) {
		case CC_CP_DIRECT:
			//		rc.flops += cpu_cc_direct(params_ptr);
			//		rc.flops += cpu_cp_direct(params_ptr);
			break;
		case CC_CP_EWALD:
			if (multis.size()) {
				//			send_ewald_to_gpu(params_ptr).get();
			}
			break;
		case PC_PP_DIRECT:
			//		rc.flops += cpu_pc_direct(params_ptr);
			//		rc.flops += cpu_pp_direct(params_ptr);
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
		array<hpx::future<kick_return>, NCHILD> futs;
		futs[LEFT] = children[LEFT].kick(params_ptr, try_thread, found_ewald);

//  printf("5\n");
		params.dchecks.pop_top();
		params.echecks.pop_top();
		params.L[params.depth] = L;
		futs[RIGHT] = children[RIGHT].kick(params_ptr, false, found_ewald);
		params.depth--;
		if (params.depth != depth0) {
			printf("error\n");
			abort();
		}
		return hpx::when_all(futs.begin(), futs.end()).then(
				[rc](hpx::future<std::vector<hpx::future<kick_return>>> futfut) {
					auto futs = futfut.get();
					auto rc1 = futs[LEFT].get();
					auto rc2 = futs[RIGHT].get();
					kick_return rc3 = rc;
					rc3.rung = std::max(rc1.rung, rc2.rung);
					rc3.flops += rc1.flops + rc2.flops;
					return rc3;
				});
	} else {
		int max_rung = 0;
		const auto invlog2 = 1.0f / logf(2);
		for (int k = 0; k < parts.second - parts.first; k++) {
			const auto this_rung = particles->rung(k + parts.first);
			if (this_rung >= params.rung) {
				array<float, NDIM> g;
				float phi;
				for (int dim = 0; dim < NDIM; dim++) {
					const auto x2 = pos[dim];
					const auto x1 = particles->pos(dim, k + parts.first);
					dx[dim] = distance(x1, x2);
				}
				shift_expansion(L, g, phi, dx);
				//		int dummy;
				//		printf( "%li\n", params.stack_top - (uintptr_t)(&dummy));
				for (int dim = 0; dim < NDIM; dim++) {
					F[dim][k] += g[dim];
				}
#ifdef TEST_FORCE
				for (int dim = 0; dim < NDIM; dim++) {
					particles->force(dim, k + parts.first) = F[dim][k];
				}
#endif
				float dt = params.t0 / (1 << this_rung);
				if (!params.t0) {
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
				dt = std::min(std::sqrt(params.scale * params.eta / fmag), params.t0);
				int new_rung = std::max(std::max(int(std::ceil(std::log(params.t0 / dt) * invlog2)), this_rung - 1),
						params.rung);
				dt = params.t0 / (1 << new_rung);
				for (int dim = 0; dim < NDIM; dim++) {
					particles->vel(dim, k + parts.first) += 0.5 * dt * F[dim][k];
				}
				max_rung = std::max(max_rung, new_rung);
				particles->set_rung(new_rung, k + parts.first);
			}
		}
		rc.rung = max_rung;
		return hpx::make_ready_future(rc);
	}
}

lockfree_queue<gpu_kick, GPU_QUEUE_SIZE> tree::gpu_queue;
lockfree_queue<gpu_ewald, GPU_QUEUE_SIZE> tree::gpu_ewald_queue;
std::atomic<bool> tree::daemon_running(false);
std::atomic<bool> tree::shutdown_daemon(false);

using cuda_exec_return =
std::pair<std::function<bool()>, kick_return*>;

cuda_exec_return cuda_execute_kick_kernel(kick_params_type **params, int grid_size);
void cuda_execute_ewald_cc_kernel(kick_params_type *params_ptr);

void tree::gpu_daemon() {
	static bool first_call = true;
	static std::vector<std::function<bool()>> completions;
	static timer timer;
	static bool skip;
	static bool ewald_skip;
	static double wait_time = 1.0e-3;
	static int min_ewald;
	if (first_call) {
		printf("Starting gpu daemon\n");
		timer.reset();
		timer.start();
		first_call = false;
		min_ewald = KICK_EWALD_GRID_SIZE;
	}
	timer.stop();
	if (timer.read() > wait_time) {
		timer.reset();
		timer.start();
		bool found_ewald = false;
		if (gpu_ewald_queue.size() >= min_ewald) {
			while (gpu_ewald_queue.size() >= min_ewald) {
				int grid_size = std::min(KICK_EWALD_GRID_SIZE, (int) gpu_ewald_queue.size());
				min_ewald = KICK_EWALD_GRID_SIZE;
				found_ewald = true;
				auto promises = std::make_shared<std::vector<hpx::lcos::local::promise<int32_t>>>();
				unified_allocator all_params_alloc;
				kick_params_type **all_params = (kick_params_type**) all_params_alloc.allocate(
						grid_size * sizeof(kick_params_type*));
				for (int i = 0; i < grid_size; i++) {
					auto tmp = gpu_ewald_queue.pop();
					all_params[i] = tmp.params;
					promises->push_back(std::move(tmp.promise));
				}
				printf("Executing %i ewald blocks\n", grid_size);
				auto exec_ret = cuda_execute_ewald_kernel(all_params, grid_size);
				completions.push_back(std::function<bool()>([=]() {
					if (exec_ret()) {
						printf("Done executing %i ewald blocks\n", grid_size);
						for (int i = 0; i < grid_size; i++) {
							(*promises)[i].set_value(all_params[i]->flops);
						}
						unified_allocator all_params_alloc;
						all_params_alloc.deallocate(all_params);
						return true;
					} else {
						return false;
					}
				}));
			}
		} else if (gpu_queue.size() == cuda_block_count) {
			int grid_size = gpu_queue.size();
			auto promises = std::make_shared<std::vector<hpx::lcos::local::promise<kick_return>>>();
			auto deleters = std::make_shared<std::vector<std::function<void()>> >();
			unified_allocator calloc;
			kick_params_type *all_params = (kick_params_type*) calloc.allocate(grid_size * sizeof(kick_params_type));
			auto stream = get_stream();
			for (int i = 0; i < grid_size; i++) {
				auto tmp = gpu_queue.pop();
				//      particles->prefetch(tmp.parts.first,tmp.parts.second,stream);
				deleters->push_back(tmp.params->dchecks.to_device(stream));
				deleters->push_back(tmp.params->echecks.to_device(stream));
				memcpy(all_params + i, tmp.params, sizeof(kick_params_type));
				auto tmpparams = tmp.params;
				deleters->push_back([tmpparams]() {
					kick_params_alloc.deallocate(tmpparams);
				});
				promises->push_back(std::move(tmp.promise));
			}
			deleters->push_back([calloc, all_params]() {
				unified_allocator calloc;
				calloc.deallocate(all_params);
			});
			void *test;
			//        CUDA_CHECK(cudaMemPrefetchAsync(all_params, grid_size * sizeof(kick_params_type), 0, stream));
			printf("Executing %i blocks\n", grid_size);
			auto exec_ret = cuda_execute_kick_kernel(all_params, grid_size, stream);
			tmp_tm.stop();
			printf("%e\n", tmp_tm.read());
			completions.push_back(std::function<bool()>([=]() {
				if (exec_ret.first()) {
					for (auto &d : *deleters) {
						d();
					}
					for (int i = 0; i < grid_size; i++) {
						(*promises)[i].set_value(exec_ret.second[i]);
					}
					unified_allocator alloc;
					alloc.deallocate(exec_ret.second);
					printf("Done executing %li blocks\n", promises->size());
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

hpx::future<kick_return> tree::send_kick_to_gpu(kick_params_type *params) {
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
	new_params->t0 = params->t0;
	new_params->first = params->first;
	new_params->tptr = me;
	new_params->dchecks = params->dchecks.copy_top();
	new_params->echecks = params->echecks.copy_top();
	new_params->L[params->depth] = params->L[params->depth];
	new_params->Lpos[params->depth] = params->Lpos[params->depth];
	new_params->depth = params->depth;
	new_params->theta = params->theta;
	new_params->eta = params->eta;
	new_params->scale = params->scale;
	new_params->rung = params->rung;
	new_params->cuda_cutoff = params->cuda_cutoff;
	gpu.params = new_params;
	auto fut = gpu.promise.get_future();
	gpu.parts = parts;
	gpu_queue.push(std::move(gpu));

	return std::move(fut);

}

int tree::compute_block_count(size_t cutoff) {
	if (parts.second - parts.first <= cutoff) {
		return 1;
	} else {
		auto left = ((tree*) children[LEFT])->compute_block_count(cutoff);
		auto right = ((tree*) children[RIGHT])->compute_block_count(cutoff);
		return left + right;
	}
}

hpx::future<int32_t> tree::send_ewald_to_gpu(kick_params_type *params) {
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

int tree::cpu_cc_direct(kick_params_type *params_ptr) {
	kick_params_type &params = *params_ptr;
	auto &L = params.L[params.depth];
	auto &multis = params.multi_interactions;
	int flops = 0;
	array<simd_int, NDIM> X;
	array<simd_int, NDIM> Y;
	array<simd_float, NDIM> dX;
	expansion<simd_float> D;
	multipole_type<simd_float> M;
	expansion<simd_float> Lacc;
	for (int dim = 0; dim < NDIM; dim++) {
		X[dim] = fixed<int>(pos[dim]).raw();
	}
	if (multis.size()) {
		for (int i = 0; i < LP; i++) {
			Lacc[i] = 0.f;
		}
		const auto cnt1 = multis.size();
		for (int j = 0; j < cnt1; j += simd_float::size()) {
			for (int k = 0; k < simd_float::size(); k++) {
				if (j + k < cnt1) {
					for (int dim = 0; dim < NDIM; dim++) {
						Y[dim][k] = fixed<int>(((const tree*) multis[j + k])->pos[dim]).raw();
					}
					for (int i = 0; i < MP; i++) {
						M[i][k] = (((const tree*) multis[j + k])->multi)[i];
					}
				} else {
					for (int dim = 0; dim < NDIM; dim++) {
						Y[dim][k] = fixed<int>(((const tree*) multis[cnt1 - 1])->pos[dim]).raw();
					}
					for (int i = 0; i < MP; i++) {
						M[i][k] = 0.f;
					}
				}
			}
			for (int dim = 0; dim < NDIM; dim++) {
				dX[dim] = simd_float(X[dim] - Y[dim]) * simd_float(fixed2float);
			}
			green_direct(D, dX);
			auto tmp = multipole_interaction(Lacc, M, D);
			if (j == 0) {
				flops = 3 + tmp;
			}
		}
		flops *= cnt1;
		for (int k = 0; k < simd_float::size(); k++) {
			for (int i = 0; i < LP; i++) {
				L[i] += Lacc[i][k];
				flops++;
			}
		}
	}
	return flops;
}

int tree::cpu_cp_direct(kick_params_type *params_ptr) {
	kick_params_type &params = *params_ptr;
	auto &L = params.L[params.depth];
	int nparts = parts.second - parts.first;
	static thread_local std::array<std::vector<fixed32>, NDIM> sources;
	auto &partis = params.part_interactions;
	for (int dim = 0; dim < NDIM; dim++) {
		sources[dim].resize(0);
	}
	for (int k = 0; k < partis.size(); k++) {
		const auto& other_parts = ((tree*) partis[k])->parts;
		for (size_t l = other_parts.first; l < other_parts.second; l++) {
			for (int dim = 0; dim < NDIM; dim++) {
				sources[dim].push_back(particles->pos(dim, l));
			}
		}
	}
	array<simd_int, NDIM> X;
	array<simd_int, NDIM> Y;
	array<simd_float, NDIM> dX;
	expansion<simd_float> D;
	simd_float M;
	expansion<simd_float> Lacc;
	for (int dim = 0; dim < NDIM; dim++) {
		X[dim] = fixed<int>(pos[dim]).raw();
	}
	for (int i = 0; i < LP; i++) {
		Lacc[i] = 0.f;
	}
	const auto cnt1 = sources[0].size();
	for (int j = 0; j < cnt1; j += simd_float::size()) {
		for (int k = 0; k < simd_float::size(); k++) {
			if (j + k < cnt1) {
				for (int dim = 0; dim < NDIM; dim++) {
					Y[dim][k] = sources[dim][j + k].raw();
				}
				for (int i = 0; i < MP; i++) {
					M[k] = 1.f;
				}
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
			dX[dim] = simd_float(X[dim] - Y[dim]) * simd_float(fixed2float);
		}
		green_direct(D, dX);
		multipole_interaction(Lacc, M, D);
	}
	for (int k = 0; k < simd_float::size(); k++) {
		for (int i = 0; i < LP; i++) {
			L[i] += Lacc[i][k];
		}
	}
	return 0;
}

int tree::cpu_pp_direct(kick_params_type *params_ptr) {
	kick_params_type &params = *params_ptr;
	auto &L = params.L[params.depth];
	auto& F = params.F;
	int nparts = parts.second - parts.first;
	static thread_local std::array<std::vector<fixed32>, NDIM> sources;
	auto &partis = params.part_interactions;
	for (int dim = 0; dim < NDIM; dim++) {
		sources[dim].resize(0);
	}
	for (int k = 0; k < partis.size(); k++) {
		const auto& other_parts = ((tree*) partis[k])->parts;
		for (size_t l = other_parts.first; l < other_parts.second; l++) {
			for (int dim = 0; dim < NDIM; dim++) {
				sources[dim].push_back(particles->pos(dim, l));
			}
		}
	}
	simd_float mask;
	array<simd_int, NDIM> X;
	array<simd_int, NDIM> Y;
	for (int k = 0; k < nparts; k++) {
		for (int dim = 0; dim < NDIM; dim++) {
			X[dim] = particles->pos(dim, k + parts.first).raw();
		}
		if (particles->rung(k + parts.first) >= params.rung) {
			array<simd_float, NDIM> f;
			for (int j = 0; j < sources[0].size(); j += simd_float::size()) {
				for (int k = 0; k < simd_float::size(); k++) {
					if (j + k < sources[0].size()) {
						mask[k] = 1.f;
						for (int dim = 0; dim < NDIM; dim++) {
							Y[dim][k] = sources[dim][j + k].raw();
						}
					} else {
						mask[k] = 0.f;
						for (int dim = 0; dim < NDIM; dim++) {
							Y[dim][k] = sources[dim][sources[0].size() - 1].raw();
						}
					}
				}
				array<simd_float, NDIM> dX;
				for (int dim = 0; dim < NDIM; dim++) {
					dX[dim] = simd_float(X[dim] - Y[dim]) * simd_float(fixed2float);
				}
				const auto r2 = fma(dX[0], dX[0], fma(dX[1], dX[1], dX[2] * dX[2]));
				const auto rinv = simd_float(1) / sqrt(r2);
				const auto rinv3 = mask * rinv * rinv * rinv;
				for (int dim = 0; dim < NDIM; dim++) {
					f[dim] = fma(rinv3, dX[dim], f[dim]);
				}
			}
			for (int dim = 0; dim < NDIM; dim++) {
				F[dim][k] += f[dim].sum();
			}
		}
	}
	return 0;
}

int tree::cpu_pc_direct(kick_params_type *params_ptr) {
	kick_params_type &params = *params_ptr;
	auto &L = params.L[params.depth];
	auto &multis = params.multi_interactions;
	auto& F = params.F;
	int nparts = parts.second - parts.first;
	array<simd_int, NDIM> X;
	array<simd_float, NDIM> dX;
	array<simd_int, NDIM> Y;
	multipole_type<simd_float> M;
	expansion<simd_float> D;
	for (int i = 0; i < nparts; i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			X[dim] = particles->pos(dim, i + parts.first).raw();
		}
		if (particles->rung(i + parts.first) >= params.rung) {
			array<simd_float, NDIM> f;
			array<simd_float, NDIM + 1> Lacc;
			for (int i = 0; i < NDIM + 1; i++) {
				Lacc[i] = 0.f;
			}
			const auto cnt1 = multis.size();
			for (int j = 0; j < cnt1; j += simd_float::size()) {
				for (int k = 0; k < simd_float::size(); k++) {
					if (j + k < cnt1) {
						for (int dim = 0; dim < NDIM; dim++) {
							Y[dim][k] = fixed<int>(((const tree*) multis[j + k])->pos[dim]).raw();
						}
						for (int i = 0; i < MP; i++) {
							M[i][k] = (((const tree*) multis[j + k])->multi)[i];
						}
					} else {
						for (int dim = 0; dim < NDIM; dim++) {
							Y[dim][k] = fixed<int>(((const tree*) multis[cnt1 - 1])->pos[dim]).raw();
						}
						for (int i = 0; i < MP; i++) {
							M[i][k] = 0.f;
						}
					}
				}
				for (int dim = 0; dim < NDIM; dim++) {
					dX[dim] = simd_float(X[dim] - Y[dim]) * simd_float(fixed2float);
				}
				green_direct(D, dX);
				multipole_interaction(Lacc, M, D);
			}
			for (int dim = 0; dim < NDIM; dim++) {
				F[dim][i] += Lacc[1 + dim].sum();
			}
		}
	}
	return 0;
}
/*
 int tree::cpu_cc_ewald(kick_params_type *params_ptr) {
 //printf("Executing ewald\n");
 kick_params_type &params = *params_ptr;
 auto &L = params.L[params.depth];
 auto &multis = params.multi_interactions;
 int flops = 0;
 if (multis.size()) {
 array<fixed32, NDIM> X, Y;
 multipole_type<float> M;
 expansion<float> Lacc;
 Lacc = 0;
 const auto cnt1 = multis.size();
 for (int dim = 0; dim < NDIM; dim++) {
 X[dim] = pos[dim];
 }
 array<float, NDIM> dX;
 for (int j = 0; j < cnt1; j++) {
 for (int dim = 0; dim < NDIM; dim++) {
 Y[dim] = ((const tree*) multis[j])->pos[dim];
 }
 M = (((const tree*) multis[j])->multi);
 for (int dim = 0; dim < NDIM; dim++) {
 dX[dim] = distance(X[dim], Y[dim]);
 }
 expansion<float> D;
 green_ewald(D, dX, *real_indices_ptr, *four_indices_ptr, *periodic_parts_ptr);
 auto tmp = multipole_interaction(Lacc, M, D);
 if (j == 0) {
 flops = 3 + tmp;
 }
 }
 flops *= cnt1;
 for (int i = 0; i < LP; i++) {
 L[i] += Lacc[i];
 flops++;
 }
 }
 return flops;
 }*/

