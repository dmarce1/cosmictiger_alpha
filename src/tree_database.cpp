
#include <cosmictiger/tree.hpp>
#include <cosmictiger/tree_database.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/memory.hpp>


void tree_database_destroy_neighbors(unrolled<tree_ptr>& neighbors) {
	neighbors.~unrolled();
}
