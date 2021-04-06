#include <cosmictiger/groups.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/timer.hpp>

struct gpu_groups {
	group_param_type* params;
	size_t part_begin;
	hpx::lcos::local::promise<bool> promise;
};

static std::vector<gpu_groups> gpu_queue;
static mutex_type mutex;
static std::atomic<size_t> parts_covered;

hpx::future<bool> tree_ptr::find_groups(group_param_type* params_ptr, bool thread) {
	group_param_type &params = *params_ptr;
	static std::atomic<int> used_threads(0);
	static unified_allocator alloc;
	static const bool use_cuda = global().opts.cuda;
	const auto myparts = get_parts();
	int counter = 0;
	if (use_cuda && myparts.second - myparts.first <= params.block_cutoff) {
		group_param_type *new_params;
		new_params = (group_param_type*) alloc.allocate(sizeof(group_param_type));
		new (new_params) group_param_type();
		*new_params = *params_ptr;
		hpx::future<bool> fut;
		{
			std::lock_guard<mutex_type> lock(mutex);
			const int gsz = gpu_queue.size();
			gpu_queue.resize(gsz + 1);
			fut = gpu_queue[gsz].promise.get_future();
			gpu_queue[gsz].params = new_params;
			gpu_queue[gsz].part_begin = myparts.first;
		}
		parts_covered += myparts.second - myparts.first;
		if (parts_covered == params.parts.size()) {
			static char waiting_char[4] = { '-', '\\', '|', '/' };
			std::sort(gpu_queue.begin(), gpu_queue.end(), [](const gpu_groups& a,const gpu_groups& b) {
				return a.part_begin < b.part_begin;
			});
			std::vector<std::function<bool()>> completions;
			const int max_oc = 16 * global().cuda.devices[0].multiProcessorCount;
			static int num_active;
			static int num_completed;
			int max_active = 2 * max_oc;
			timer tm;
			num_completed = 0;
			num_active = 0;
			while (gpu_queue.size() || completions.size()) {
				tm.stop();
				if (tm.read() > 0.05) {
					counter ++;
					printf("%c %i in queue: %i active: %i complete                        \r", waiting_char[counter % 4], gpu_queue.size(), num_active,
							num_completed);
					counter++;
					tm.reset();
				}
				tm.start();
				if (gpu_queue.size() && num_active + max_oc <= max_active) {
					cudaStream_t stream = get_stream();
					auto this_kernel = std::make_shared<std::vector<gpu_groups>>();
					int sz = std::min(max_oc, (int) gpu_queue.size());
					for (int i = 0; i < sz; i++) {
						this_kernel->push_back(std::move(gpu_queue.back()));
						gpu_queue.pop_back();
					}
					group_param_type** all_params;
					unified_allocator alloc;
					all_params = (group_param_type**) alloc.allocate(sizeof(group_param_type*) * this_kernel->size());
					std::vector<std::function<void()>> deleters;
					for (int i = 0; i < this_kernel->size(); i++) {
//					printf( "Setting up %i of %i\n", i, this_kernel->size());
						all_params[i] = (*this_kernel)[i].params;
						deleters.push_back(all_params[i]->next_checks.to_device(stream));
						deleters.push_back(all_params[i]->opened_checks.to_device(stream));
						deleters.push_back(all_params[i]->checks.to_device(stream));
					}
					num_active += this_kernel->size();
					auto cfunc = call_cuda_find_groups(all_params, this_kernel->size(), stream);
					completions.push_back([cfunc,deleters, all_params, this_kernel, stream]() {
						auto rc = cfunc();
						if( rc.size()) {
							num_completed += this_kernel->size();
							num_active -= this_kernel->size();
							for( int i = 0; i < deleters.size(); i++) {
								deleters[i]();
							}
							for( int i = 0; i < rc.size(); i++) {
								(*this_kernel)[i].promise.set_value(rc[i]);
							}
							unified_allocator alloc;
							alloc.deallocate(all_params);
							cleanup_stream(stream);
						}
						return rc.size();
					});
				}
				while (completions.size()) {
					int i = 0;
					while (i < completions.size()) {
						if (completions[i]()) {
							completions[i] = completions.back();
							completions.pop_back();
						} else {
							i++;
						}
					}
				}
			}
		}
		return std::move(fut);

	} else {
		if (thread) {
			const int max_threads = OVERSUBSCRIPTION * hpx::threads::hardware_concurrency();
			if (used_threads++ > max_threads) {
				used_threads--;
				thread = false;
			}
		}
		if (thread) {
			group_param_type *new_params;
			new_params = (group_param_type*) alloc.allocate(sizeof(group_param_type));
			new (new_params) group_param_type();
			*new_params = *params_ptr;
			auto func = [new_params]() {
				auto rc = ::find_groups(new_params);
				used_threads--;
				new_params->group_param_type::~group_param_type();
				alloc.deallocate(new_params);
				return rc;
			};
			auto fut = hpx::async(std::move(func));
			return std::move(fut);
		} else {
			auto fut = ::find_groups(params_ptr);
			return fut;
		}
	}
}

hpx::future<bool> find_groups(group_param_type* params_ptr) {
	group_param_type& params = *params_ptr;
	auto& parts = params.parts;
	tree_ptr self = params.self;
	if (params.depth == 0) {
		parts_covered = 0;
		if (params.first_round) {
			parts.init_groups();
		}
		gpu_queue.resize(0);
		size_t dummy, total_mem;
		CUDA_CHECK(cudaMemGetInfo(&dummy, &total_mem));
		total_mem /= 8;
		size_t used_mem = (sizeof(group_t) + sizeof(fixed32) * NDIM) * parts.size();
		double oversubscription = std::max(2.0, (double) used_mem / total_mem);
		const int block_count = oversubscription * global().cuda_kick_occupancy
				* global().cuda.devices[0].multiProcessorCount + 0.5;
		const auto myparts = self.get_parts();
		params.block_cutoff = std::max((myparts.second - myparts.first) / block_count, (size_t) 1);
	}

	auto& checks = params.checks;
	auto& next_checks = params.next_checks;
	auto& opened_checks = params.opened_checks;

	const auto myrange = self.get_range();
	const auto iamleaf = self.is_leaf();
	opened_checks.resize(0);
	do {
		next_checks.resize(0);
		for (int i = 0; i < checks.size(); i++) {
			const auto other_range = checks[i].get_range();
			if (myrange.intersects(other_range)) {
				if (checks[i].is_leaf()) {
					opened_checks.push_back(checks[i]);
				} else {
					next_checks.push_back(checks[i]);
				}
			}
		}
		checks.resize(NCHILD * next_checks.size());
		for (int i = 0; i < next_checks.size(); i++) {
			const auto children = next_checks[i].get_children();
			checks[NCHILD * i + LEFT] = children[LEFT];
			checks[NCHILD * i + RIGHT] = children[RIGHT];
		}
	} while (iamleaf && checks.size());
	for (int i = 0; i < opened_checks.size(); i++) {
		checks.push(opened_checks[i]);
	}
	if (iamleaf) {
		const auto myparts = self.get_parts();
		const auto linklen2 = sqr(params.link_len);
		bool found_link;
		found_link = false;
		for (int i = 0; i < checks.size(); i++) {
			const auto other_parts = checks[i].get_parts();
			for (int j = myparts.first; j != myparts.second; j++) {
				for (int k = other_parts.first; k != other_parts.second; k++) {
					float dx0, dx1, dx2;
					dx0 = distance(parts.pos(0, j), parts.pos(0, k));
					dx1 = distance(parts.pos(1, j), parts.pos(1, k));
					dx2 = distance(parts.pos(2, j), parts.pos(2, k));
					const float dist2 = fma(dx0, dx0, fma(dx1, dx1, sqr(dx2)));
					if (dist2 < linklen2 && dist2 != 0.0) {
						if (parts.group(j) == NO_GROUP) {
							parts.group(j) = j;
						}
						if (parts.group(k) == NO_GROUP) {
							parts.group(k) = k;
						}
						auto& id1 = parts.group(k);
						auto& id2 = parts.group(j);
						const auto shared_id = std::min(id1, id2);
						if (id1 != shared_id || id2 != shared_id) {
							found_link = true;
						}
						id1 = id2 = shared_id;
					}
				}
			}
		}
		return hpx::make_ready_future(found_link);
	} else {
		std::array<hpx::future<bool>, NCHILD> futs;
		bool found_link;
		auto mychildren = self.get_children();
		params.checks.push_top();
		params.self = mychildren[LEFT];
		params.depth++;
		futs[LEFT] = mychildren[LEFT].find_groups(params_ptr, true);
		params.checks.pop_top();
		params.self = mychildren[RIGHT];
		futs[RIGHT] = mychildren[RIGHT].find_groups(params_ptr, false);
		params.depth--;
		return hpx::when_all(futs.begin(), futs.end()).then([](hpx::future<std::vector<hpx::future<bool>>> futfut) {
			auto futs = futfut.get();
			const bool rcr = futs[RIGHT].get();
			return futs[LEFT].get() || rcr;
		});
	}
}
