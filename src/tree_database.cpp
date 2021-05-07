#include <cosmictiger/tree.hpp>
#include <cosmictiger/tree_database.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/memory.hpp>
#include <cosmictiger/hpx.hpp>

void tree_data_initialize_kick();
void tree_data_initialize_groups();
void tree_data_free_all_cu();
void tree_data_set_groups_cu();

HPX_PLAIN_ACTION (tree_data_initialize_kick);
HPX_REGISTER_BROADCAST_ACTION_DECLARATION (tree_data_initialize_kick_action);
HPX_REGISTER_BROADCAST_ACTION (tree_data_initialize_kick_action);

HPX_PLAIN_ACTION (tree_data_initialize_groups);
HPX_REGISTER_BROADCAST_ACTION_DECLARATION (tree_data_initialize_groups_action);
HPX_REGISTER_BROADCAST_ACTION (tree_data_initialize_groups_action);

HPX_PLAIN_ACTION (tree_data_free_all_cu);
HPX_REGISTER_BROADCAST_ACTION_DECLARATION (tree_data_free_all_cu_action);
HPX_REGISTER_BROADCAST_ACTION (tree_data_free_all_cu_action);

HPX_PLAIN_ACTION (tree_data_set_groups_cu);
HPX_REGISTER_BROADCAST_ACTION_DECLARATION (tree_data_set_groups_cu_action);
HPX_REGISTER_BROADCAST_ACTION (tree_data_set_groups_cu_action);

void tree_data_initialize(tree_use_type use_type) {
	if (hpx_rank == 0) {
		if (use_type == KICK) {
			hpx::lcos::broadcast < tree_data_initialize_kick_action > (hpx_localities()).get();
		} else {
			hpx::lcos::broadcast < tree_data_initialize_groups_action > (hpx_localities()).get();
		}
	}
}

void tree_data_free_all() {
	hpx::lcos::broadcast < tree_data_free_all_cu_action > (hpx_localities()).get();
}

void tree_data_set_groups() {
	hpx::lcos::broadcast < tree_data_set_groups_cu_action > (hpx_localities()).get();
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

