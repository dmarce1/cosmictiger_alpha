#include <cosmictiger/tree.hpp>
#include <cosmictiger/tree_database.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/memory.hpp>
#include <cosmictiger/hpx.hpp>

HPX_PLAIN_ACTION(tree_data_initialize);
HPX_PLAIN_ACTION(tree_data_free_all);

#define TREE_CACHE_SIZE 1024

struct tree_cache_entry {
	hpx::shared_future<void> ready_fut;
	std::vector<tree_node_t> data;
};

struct tree_hash_lo {
	size_t operator()(tree_ptr ptr) const {
		const int i = ptr.dindex / TREE_CACHE_LINE_SIZE * hpx_size() + hpx_rank();
		return (i) % TREE_CACHE_SIZE;
	}
};

struct tree_hash_hi {
	size_t operator()(tree_ptr ptr) const {
		const int i = ptr.dindex / TREE_CACHE_LINE_SIZE * hpx_size() + hpx_rank();
		return (i) / TREE_CACHE_SIZE;
	}
};


using tree_cache_map_type =std::unordered_map<tree_ptr, std::shared_ptr<tree_cache_entry>, tree_hash_hi>;
static std::array<mutex_type, TREE_CACHE_SIZE> mutexes;
static std::array<tree_cache_map_type, TREE_CACHE_SIZE> caches;
static std::unordered_map<tree_ptr, int, tree_hash> tree_map;

tree_ptr tree_data_global_to_local(tree_ptr global) {
	tree_ptr local;
	local.rank = hpx_rank();
	if (global.rank != local.rank) {
		auto iter = tree_map.find(global);
		if (iter == tree_map.end()) {
			ERROR()
			;
		}
		local.dindex = iter->second;
	} else {
		local.dindex = global.dindex;
	}
	return local;
}

void tree_data_map_global_to_local() {
	const int nthreads = hardware_concurrency();
	static spinlock_type mutex;
	std::vector<hpx::future<void>> futs;
	tree_map = decltype(tree_map)();
	for (int proc = 0; proc < nthreads; proc++) {
		const auto func = [nthreads,proc]() {
			tree_allocator tree_alloc;
			const int start = proc * TREE_CACHE_SIZE / nthreads;
			const int stop = (proc + 1) * TREE_CACHE_SIZE / nthreads;
			for (int i = start; i < stop; i++) {
				auto& cache = caches[i];
				for (auto k = cache.begin(); k != cache.end(); k++) {
					const auto& line = k->second;
					for (int l = 0; l < line->data.size(); l++) {
						int index = tree_alloc.allocate();
						const auto& entry = line->data[l];
						cpu_tree_data_.active_nodes[index] = entry.active_nodes;
						cpu_tree_data_.active_parts[index] = entry.active_parts;
						if (cpu_tree_data_.multi) {
							cpu_tree_data_.multi[index].multi = entry.multi;
							cpu_tree_data_.multi[index].pos = entry.pos;
						}
						cpu_tree_data_.data[index].pos = entry.pos;
						cpu_tree_data_.data[index].radius = entry.radius;
						if (cpu_tree_data_.ranges) {
							cpu_tree_data_.ranges[index] = entry.ranges;
						}
						cpu_tree_data_.parts[index] = entry.parts;
						cpu_tree_data_.proc_range[index] = entry.proc_range;
						cpu_tree_data_.local_root[index] = entry.local_root;
						cpu_tree_data_.data[index].children = entry.children;
						tree_ptr global_ptr;
						global_ptr.dindex = k->first.dindex + l;
						global_ptr.rank = k->first.rank;
						assert(global_ptr.rank !=hpx_rank());
						std::lock_guard<spinlock_type> lock(mutex);
						tree_map[global_ptr] = index;
					}
				}
				cache = tree_cache_map_type();
			}};
		futs.push_back(hpx::async(func));
	}
	hpx::wait_all(futs.begin(), futs.end());
	futs.resize(0);
	for (int proc = 0; proc < nthreads; proc++) {
		const auto func = [nthreads,proc]() {
			tree_allocator tree_alloc;
			const int start = proc * cpu_tree_data_.ntrees / nthreads;
			const int stop = (proc + 1) * cpu_tree_data_.ntrees / nthreads;
			for (int i = start; i < stop; i++) {
				auto& children = cpu_tree_data_.data[i].children;
				const auto& proc_range = cpu_tree_data_.proc_range[i];
				if( proc_range.second - proc_range.first == 1 && children[0].dindex != -1) {
					for( int ci = 0; ci < NCHILD; ci++) {
						const auto iter = tree_map.find(children[ci]);
						if( iter != tree_map.end()) {
							children[ci].dindex = iter->second;
							children[ci].rank = hpx_rank();
						}
					}
				}
			}};
		futs.push_back(hpx::async(func));
	}
	hpx::wait_all(futs.begin(), futs.end());
}

std::vector<tree_node_t> tree_data_fetch_cache_line(int index);

static int cache_line_index(int index) {
	return index - index % TREE_CACHE_LINE_SIZE;

}

HPX_PLAIN_ACTION(tree_data_fetch_cache_line);

void tree_data_global_to_local(stack_vector<tree_ptr>& stack) {
	for (int i = 0; i < stack.size(); i++) {
		stack[i] = tree_data_global_to_local(stack[i]);
	}
}

void tree_data_free_cache() {
	for (int i = 0; i < TREE_CACHE_SIZE; i++) {
		caches[i] = tree_cache_map_type();
	}
}

tree_node_t& tree_data_load_cache(tree_ptr ptr) {
	if (ptr.rank >= hpx_size()) {
		printf("Rank out of range %i\n", ptr.rank);
		ERROR()
		;
	}
	if (ptr.dindex > cpu_tree_data_.ntrees) {
		ERROR()
		;
	}
	static const tree_hash_lo hashlo;
	tree_ptr line_ptr;
	line_ptr.dindex = cache_line_index(ptr.dindex);
	line_ptr.rank = ptr.rank;
	const int loindex = hashlo(line_ptr);
	auto& mutex = mutexes[loindex];
	auto& cache = caches[loindex];
	std::unique_lock<mutex_type> lock(mutex);
	auto i = cache.find(line_ptr);
	if (i == cache.end()) {
		auto& entry = cache[line_ptr];
		auto prms = std::make_shared<hpx::lcos::local::promise<void>>();
		entry = std::make_shared<tree_cache_entry>();
		entry->ready_fut = prms->get_future();
		lock.unlock();
		hpx::apply([prms,i,loindex,line_ptr]() {
			tree_data_fetch_cache_line_action act;
			auto line = act(hpx_localities()[line_ptr.rank], line_ptr.dindex);
			auto& mutex = mutexes[loindex];
			auto& cache = caches[loindex];
			std::unique_lock<mutex_type> lock(mutex);
			cache[line_ptr]->data = std::move(line);
			lock.unlock();
			prms->set_value();
		});
		lock.lock();
		i = cache.find(line_ptr);
	}
	auto entry = i->second;
	lock.unlock();
	entry->ready_fut.get();
	return entry->data[ptr.dindex - line_ptr.dindex];
}

multipole tree_data_read_cache_multi(tree_ptr ptr) {
	auto& dat = tree_data_load_cache(ptr);
	return dat.multi;
}

array<fixed32, NDIM> tree_data_read_cache_pos(tree_ptr ptr) {
	auto& dat = tree_data_load_cache(ptr);
	return dat.pos;
}

float tree_data_read_cache_radius(tree_ptr ptr) {
	auto& dat = tree_data_load_cache(ptr);
	return dat.radius;

}

part_iters tree_data_read_cache_parts(tree_ptr ptr) {
	auto& dat = tree_data_load_cache(ptr);
	return dat.parts;
}

bool tree_data_read_cache_local_root(tree_ptr ptr) {
	auto& dat = tree_data_load_cache(ptr);
	return dat.local_root;
}

size_t tree_data_read_cache_active_nodes(tree_ptr ptr) {
	auto& dat = tree_data_load_cache(ptr);
	return dat.active_nodes;
}

pair<int, int> tree_data_read_cache_proc_range(tree_ptr ptr) {
	auto& dat = tree_data_load_cache(ptr);
	return dat.proc_range;
}

array<tree_ptr, NCHILD> tree_data_read_cache_children(tree_ptr ptr) {
	auto& dat = tree_data_load_cache(ptr);
	return dat.children;
}

range tree_data_read_cache_range(tree_ptr ptr) {
	auto& dat = tree_data_load_cache(ptr);
	return dat.ranges;
}

size_t tree_data_read_cache_active_parts(tree_ptr ptr) {
	auto& dat = tree_data_load_cache(ptr);
	return dat.active_parts;
}

std::vector<tree_node_t> tree_data_fetch_cache_line(int index) {
	std::vector<tree_node_t> line;
	index = cache_line_index(index);
	const int start = index;
	const int stop = index + TREE_CACHE_LINE_SIZE;
	for (int i = start; i < stop; i++) {
		tree_node_t entry;
		entry.active_nodes = cpu_tree_data_.active_nodes[i];
		entry.active_parts = cpu_tree_data_.active_parts[i];
		if (cpu_tree_data_.multi) {
			entry.multi = cpu_tree_data_.multi[i].multi;
			entry.use = TREE_KICK;
		} else {
			entry.use = TREE_GROUPS;
		}
		entry.pos = cpu_tree_data_.multi[i].pos;
		entry.radius = cpu_tree_data_.data[i].radius;
		if (cpu_tree_data_.ranges) {
			entry.ranges = cpu_tree_data_.ranges[i];
		}
		entry.parts = cpu_tree_data_.parts[i];
		entry.proc_range = cpu_tree_data_.proc_range[i];
		entry.local_root = cpu_tree_data_.local_root[i];
		entry.children = cpu_tree_data_.data[i].children;
		line.push_back(entry);
	}
	return line;
}

void tree_data_initialize(tree_use_type type) {
	std::vector<hpx::future<void>> futs;
	if (hpx_rank() == 0) {
		for (int i = 1; i < hpx_size(); i++) {
			futs.push_back(hpx::async<tree_data_initialize_action>(hpx_localities()[i], type));
		}
	}
	if (type == TREE_KICK) {
		tree_data_initialize_kick();
	} else {
		tree_data_initialize_groups();
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void tree_data_free_all() {
	std::vector<hpx::future<void>> futs;
	if (hpx_rank() == 0) {
		for (int i = 1; i < hpx_size(); i++) {
			futs.push_back(hpx::async<tree_data_free_all_action>(hpx_localities()[i]));
		}
	}
	tree_data_free_all_cu();
	hpx::wait_all(futs.begin(), futs.end());
}
