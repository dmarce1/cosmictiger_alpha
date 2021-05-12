#include <cosmictiger/tree.hpp>
#include <cosmictiger/tree_database.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/memory.hpp>
#include <cosmictiger/hpx.hpp>
#include <unordered_map>

#define TREE_CACHE_LINE_SIZE 64
#define TREE_CACHE_SIZE 1024

static tree_use_type tree_type;

struct tree_ptr_hi_hash {
	size_t operator()(const tree_ptr& ptr) const {
		return (size_t(ptr.dindex) * size_t(hpx_size()) + size_t(ptr.rank)) / size_t(TREE_CACHE_SIZE);
	}
};

struct tree_ptr_lo_hash {
	size_t operator()(const tree_ptr& ptr) const {
		return (size_t(ptr.dindex) * size_t(hpx_size()) + size_t(ptr.rank)) % size_t(TREE_CACHE_SIZE);
	}
};

struct cache_entry_t {
	std::shared_ptr<hpx::shared_future<void>> fut;
	std::shared_ptr<tree_database_t> data;
};

using cache_type = std::unordered_map<tree_ptr, cache_entry_t, tree_ptr_hi_hash>;
static array<mutex_type, TREE_CACHE_SIZE> mutexes;
static array<cache_type, TREE_CACHE_SIZE> tree_caches;

void tree_data_initialize_kick();
void tree_data_initialize_groups();
void tree_data_free_all_cu();
void tree_data_set_groups_cu();

HPX_PLAIN_ACTION(tree_data_initialize);

HPX_PLAIN_ACTION(tree_data_initialize_groups);
HPX_PLAIN_ACTION(tree_data_free_all_cu);
HPX_PLAIN_ACTION(tree_data_set_groups_cu);
HPX_PLAIN_ACTION(tree_cache_line_fetch);

tree_database_t allocate_cache_line();
void deallocate_cache_line(tree_database_t td);
void tree_cache_load(tree_ptr ptr);

void tree_cache_compute_indices(tree_ptr& base, int& offset, tree_ptr ptr) {
	int base_index = ptr.dindex - (ptr.dindex % TREE_CACHE_LINE_SIZE);
	offset = ptr.dindex - base_index;
	base.dindex = base_index;
	base.rank = ptr.rank;
}

tree_ptr tree_cache_compute_base(tree_ptr ptr) {
	tree_ptr base;
	assert(ptr.dindex >= 0);
	assert(ptr.dindex < cpu_tree_data().ntrees);
	assert(ptr.rank >= 0);
	assert(ptr.rank < hpx_size());
	int base_index = ptr.dindex - (ptr.dindex % TREE_CACHE_LINE_SIZE);
	base.dindex = base_index;
	base.rank = ptr.rank;
	return base;
}

int tree_cache_compute_base_index(int dindex) {
	return dindex - (dindex % TREE_CACHE_LINE_SIZE);
}

tree_database_parcel_t tree_cache_line_fetch(int index) {
	printf( "Fetching line\n");
	index = tree_cache_compute_base_index(index);
	tree_database_t db = allocate_cache_line();
	for (int dindex = index; dindex < index + TREE_CACHE_LINE_SIZE; dindex++) {
		assert(dindex < cpu_tree_data().ntrees && dindex >= 0);
		int j = dindex - index;
		assert(j < TREE_CACHE_LINE_SIZE);
		db.data[j] = cpu_tree_data().data[dindex];
		db.parts[j] = cpu_tree_data().parts[dindex];
		db.active_nodes[j] = cpu_tree_data().active_nodes[dindex];
		db.active_parts[j] = cpu_tree_data().active_parts[dindex];
		db.all_local[j] = cpu_tree_data().all_local[dindex];
		if (db.ranges) {
			db.ranges[j] = cpu_tree_data().ranges[dindex];
		}
		if (db.sph_ranges) {
			db.sph_ranges[j] = cpu_tree_data().sph_ranges[dindex];
		}
		if (db.multi) {
			db.multi[j] = cpu_tree_data().multi[dindex];
		}
	}
	tree_database_parcel_t parcel;
	parcel.deleteme = true;
	parcel.data = std::move(db);
	return parcel;
}

void tree_cache_clear() {
	for (int i = 0; i < TREE_CACHE_SIZE; i++) {
		tree_caches[i] = cache_type();
	}
}

void tree_cache_get(std::shared_ptr<tree_database_t>& data_ptr, int& offset, tree_ptr ptr) {
	static tree_ptr_lo_hash key;
	tree_ptr base;
	int cache_index = key(ptr);
	tree_cache_load(ptr);
	tree_cache_compute_indices(base, offset, ptr);
	std::lock_guard<spinlock_type> lock(mutexes[cache_index]);
	data_ptr = tree_caches[cache_index][base].data;
}

void tree_cache_load(tree_ptr tptr) {
	static auto localities = hpx_localities();
	static tree_ptr_lo_hash key;
	int cache_index = key(tptr);
	std::unique_lock<spinlock_type> lock(mutexes[cache_index]);
	assert(tptr.dindex >= 0 && tptr.dindex < cpu_tree_data().ntrees);
	const auto base = tree_cache_compute_base(tptr);
	if (tree_caches[cache_index].find(base) == tree_caches[cache_index].end()) {
		auto prms = std::make_shared<hpx::lcos::local::promise<void>>();
		auto& entry = tree_caches[cache_index][base];
		entry.fut = std::make_shared<hpx::shared_future<void>>(prms->get_future());
		lock.unlock();
		hpx::async([base, cache_index, prms]() {
			tree_cache_line_fetch_action action;
			auto data = action(localities[base.rank],base.dindex);
			std::unique_lock<spinlock_type> lock(mutexes[cache_index]);
			auto& entry = tree_caches[cache_index][base];
			data.deleteme = false;
			entry.data = std::shared_ptr<tree_database_t>(new tree_database_t(std::move(data.data)), [](tree_database_t* ptr) {
						deallocate_cache_line(*ptr);
						delete ptr;
					});
			lock.unlock();
			prms->set_value();
		});
		lock.lock();
	}
	auto fut = tree_caches[cache_index][base].fut;
	lock.unlock();
	fut->get();
}

tree_database_t allocate_cache_line() {
	tree_database_t td;

	td.data = new tree_data_t[TREE_CACHE_LINE_SIZE];
	td.parts = new parts_type[TREE_CACHE_LINE_SIZE];
	td.active_nodes = new size_t[TREE_CACHE_LINE_SIZE];
	td.active_parts = new size_t[TREE_CACHE_LINE_SIZE];
	td.all_local = new bool[TREE_CACHE_LINE_SIZE];
	if (tree_type == KICK) {
		td.multi = new multipole_pos[TREE_CACHE_LINE_SIZE];
		if (global().opts.sph) {
			td.ranges = new range[TREE_CACHE_LINE_SIZE];
			td.sph_ranges = new range[TREE_CACHE_LINE_SIZE];
		} else {
			td.ranges = nullptr;
			td.sph_ranges = nullptr;
		}
	} else {
		td.sph_ranges = nullptr;
		td.multi = nullptr;
		td.ranges = new range[TREE_CACHE_LINE_SIZE];
	}
	td.chunk_size = TREE_CACHE_LINE_SIZE;
	td.nchunks = 1;
	td.ntrees = TREE_CACHE_LINE_SIZE;
	return td;
}

void deallocate_cache_line(tree_database_t td) {
	delete[] td.data;
	delete[] td.parts;
	delete[] td.active_nodes;
	delete[] td.active_parts;
	delete[] td.all_local;
	if (td.ranges) {
		delete[] td.ranges;
	}
	if (td.sph_ranges) {
		delete[] td.sph_ranges;
	}
	if (td.multi) {
		delete[] td.multi;
	}
}

void tree_data_initialize(tree_use_type use_type) {
	tree_type = use_type;
	std::vector<hpx::future<void>> futs;
	if (hpx_rank() == 0) {
		for (int i = 1; i < hpx_size(); i++) {
			if (use_type == KICK) {
				futs.push_back(hpx::async<tree_data_initialize_action>(hpx_localities()[i], use_type));
			} else {
				futs.push_back(hpx::async<tree_data_initialize_action>(hpx_localities()[i], use_type));
			}
		}
	}
	if (use_type == KICK) {
		tree_data_initialize_kick();
	} else {
		tree_data_initialize_groups();
	}
	hpx::wait_all(futs.begin(), futs.end());
	printf("tree_data_initialize done\n");
}

tree_database_t& cpu_tree_data() {
	static tree_database_t data = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1, 1, 1 };
	return data;
}

void tree_data_free_all() {
	printf("Freeing tree data for ALL\n");
	std::vector<hpx::future<void>> futs;
	for (int i = 0; i < hpx_size(); i++) {
		futs.push_back(hpx::async<tree_data_free_all_cu_action>(hpx_localities()[i]));
	}
	hpx::wait_all(futs.begin(), futs.end());

}

void tree_data_set_groups() {
	std::vector<hpx::future<void>> futs;
	for (int i = 0; i < hpx_size(); i++) {
		futs.push_back(hpx::async<tree_data_set_groups_cu_action>(hpx_localities()[i]));
	}
	hpx::wait_all(futs.begin(), futs.end());
}

tree_allocator::tree_allocator() {
	current_alloc = tree_data_allocate();
	next = current_alloc.first;
}

tree_ptr tree_allocator::allocate() {
	next++;
	if (next == current_alloc.second) {
		current_alloc = tree_data_allocate();
		next = current_alloc.first;
	}
	tree_ptr ptr;
	ptr.dindex = next;
	ptr.rank = hpx_rank();
	return ptr;
}

float tree_cache_get_radius(tree_ptr ptr) {
	std::shared_ptr<tree_database_t> data_ptr;
	int offset;
	tree_cache_get(data_ptr, offset, ptr);
	return data_ptr->data[offset].radius;
}

bool tree_cache_get_all_local(tree_ptr ptr) {
	std::shared_ptr<tree_database_t> data_ptr;
	int offset;
	tree_cache_get(data_ptr, offset, ptr);
	return data_ptr->all_local[offset];
}

array<fixed32, NDIM> tree_cache_get_pos(tree_ptr ptr) {
	std::shared_ptr<tree_database_t> data_ptr;
	int offset;
	tree_cache_get(data_ptr, offset, ptr);
	return data_ptr->data[offset].pos;
}

multipole tree_cache_get_multi(tree_ptr ptr) {
	std::shared_ptr<tree_database_t> data_ptr;
	int offset;
	tree_cache_get(data_ptr, offset, ptr);
	return data_ptr->multi[offset].multi;
}

bool tree_cache_get_isleaf(tree_ptr ptr) {
	std::shared_ptr<tree_database_t> data_ptr;
	int offset;
	tree_cache_get(data_ptr, offset, ptr);
	return data_ptr->data[offset].children[0].dindex == -1;
}

array<tree_ptr, NCHILD> tree_cache_get_children(tree_ptr ptr) {
	std::shared_ptr<tree_database_t> data_ptr;
	int offset;
	tree_cache_get(data_ptr, offset, ptr);
	return data_ptr->data[offset].children;
}

part_iters tree_cache_get_parts(tree_ptr ptr, int pi) {
	std::shared_ptr<tree_database_t> data_ptr;
	int offset;
	tree_cache_get(data_ptr, offset, ptr);
	return data_ptr->parts[offset][pi];
}

parts_type tree_cache_get_parts(tree_ptr ptr) {
	std::shared_ptr<tree_database_t> data_ptr;
	int offset;
	tree_cache_get(data_ptr, offset, ptr);
	return data_ptr->parts[offset];
}

size_t tree_cache_get_active_parts(tree_ptr ptr) {
	std::shared_ptr<tree_database_t> data_ptr;
	int offset;
	tree_cache_get(data_ptr, offset, ptr);
	return data_ptr->active_parts[offset];
}

size_t tree_cache_get_active_nodes(tree_ptr ptr) {
	std::shared_ptr<tree_database_t> data_ptr;
	int offset;
	tree_cache_get(data_ptr, offset, ptr);
	return data_ptr->active_nodes[offset];
}

range tree_cache_get_range(tree_ptr ptr) {
	std::shared_ptr<tree_database_t> data_ptr;
	int offset;
	tree_cache_get(data_ptr, offset, ptr);
	return data_ptr->ranges[offset];
}

range tree_cache_get_sph_range(tree_ptr ptr) {
	std::shared_ptr<tree_database_t> data_ptr;
	int offset;
	tree_cache_get(data_ptr, offset, ptr);
	return data_ptr->sph_ranges[offset];
}

