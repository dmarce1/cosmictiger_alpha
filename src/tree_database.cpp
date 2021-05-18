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

static std::array<mutex_type, TREE_CACHE_SIZE> mutexes;
static std::array<std::unordered_map<tree_ptr, std::shared_ptr<tree_cache_entry>, tree_hash_hi>, TREE_CACHE_SIZE> caches;

std::vector<tree_node_t> tree_data_fetch_cache_line(int index);
void tree_data_clear_cache();

static int cache_line_index(int index) {
	return index - index % TREE_CACHE_LINE_SIZE;

}

HPX_PLAIN_ACTION(tree_data_fetch_cache_line);
HPX_PLAIN_ACTION(tree_data_clear_cache);

void tree_data_clear_cache() {
	std::vector<hpx::future<void>> futs;
	if (hpx_rank() == 0) {

	}
}

tree_node_t& tree_data_load_cache(tree_ptr ptr) {
	static const tree_hash_lo hashlo;
	tree_ptr line_ptr;
	line_ptr.dindex = cache_line_index(ptr.dindex);
	line_ptr.rank = ptr.rank;
	const int loindex = hashlo(line_ptr);
	auto& mutex = mutexes[loindex];
	std::unique_lock<mutex_type> lock(mutex);
	auto& cache = caches[loindex];
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
			std::lock_guard<mutex_type> lock(mutex);
			auto& cache = caches[loindex];
			cache[line_ptr]->data = std::move(line);
			prms->set_value();
		});
		lock.lock();
		i = cache.find(line_ptr);
	}
	auto& entry = i->second;
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
