#include <cosmictiger/groups.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/timer.hpp>

struct gpu_groups {
	group_param_type* params;
	size_t part_begin;
	hpx::lcos::local::promise<bool> promise;
};

static vector<tree_ptr> group_leaves;
static mutex_type mutex;

hpx::future<void> tree_ptr::find_groups_phase1(group_param_type* params_ptr, bool thread) {
	group_param_type &params = *params_ptr;
	static std::atomic<int> used_threads(0);
	static unified_allocator alloc;
	const auto myparts = get_parts();
	int counter = 0;
	if (thread) {
		const int max_threads = 2 * hpx::threads::hardware_concurrency();
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
			auto rc = ::find_groups_phase1(new_params);
			rc.get();
			used_threads--;
			new_params->group_param_type::~group_param_type();
			alloc.deallocate(new_params);
		};
		return hpx::async(std::move(func));
	} else {
		return ::find_groups_phase1(params_ptr);
	}
}

hpx::future<void> find_groups_phase1(group_param_type* params_ptr) {
	group_param_type& params = *params_ptr;
	auto& parts = params.parts;
	tree_ptr self = params.self;
	if (params.depth == 0) {
		group_leaves.resize(0);
		parts.init_groups();
	}

	auto& param_checks = params.checks;
	auto& next_checks = params.next_checks;
	auto& opened_checks = params.opened_checks;
	auto& checks = params.tmp;

	const auto myrange = self.get_range();
	const auto iamleaf = self.is_leaf();
	checks.resize(param_checks.size());
	for( int i = 0; i < param_checks.size(); i++) {
		checks[i] = param_checks[i];
	}
	bool opened;
	opened_checks.resize(0);
	do {
		opened = false;
		next_checks.resize(0);
		for (int i = 0; i < checks.size(); i++) {
			const auto other_range = checks[i].get_range();
			if (myrange.intersects(other_range)) {
				if (checks[i].is_leaf()) {
					opened_checks.push_back(checks[i]);
				} else {
					opened = true;
					const auto children = checks[i].get_children();
					next_checks.push_back(children[LEFT]);
					next_checks.push_back(children[RIGHT]);
				}
			}
		}
		checks.swap(next_checks);
	} while (iamleaf && opened);
	for( int i = 0; i < opened_checks.size(); i++) {
		checks.push_back(opened_checks[i]);
	}
//	printf( "%i\n", checks.size());
	if (iamleaf) {
		vector<tree_ptr> neighbors(checks.size());
		for (int i = 0; i < checks.size(); i++) {
			neighbors[i] = checks[i];
		}
		self.set_neighbors(std::move(neighbors));
		std::lock_guard<mutex_type> lock(mutex);
		group_leaves.push_back(self);
		return hpx::make_ready_future();
	} else {
		param_checks.resize(checks.size());
		for( int i = 0; i < param_checks.size(); i++) {
			param_checks[i] = checks[i];
		}
		std::array<hpx::future<void>, NCHILD> futs;
		bool found_link;
		auto mychildren = self.get_children();
		params.checks.push_top();
		params.self = mychildren[LEFT];
		params.depth++;
		futs[LEFT] = mychildren[LEFT].find_groups_phase1(params_ptr, true);
		params.checks.pop_top();
		params.self = mychildren[RIGHT];
		futs[RIGHT] = mychildren[RIGHT].find_groups_phase1(params_ptr, false);
		params.depth--;
		return hpx::when_all(futs.begin(), futs.end()).then([](hpx::future<std::vector<hpx::future<void>>> futfut) {
			auto futs = futfut.get();
			futs[LEFT].get();
			futs[RIGHT].get();
		});
	}
}

bool find_groups_phase2(group_param_type* params) {
	return call_cuda_find_groups_phase2(params, group_leaves);
}
