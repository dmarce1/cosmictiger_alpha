
#include <cosmictiger/groups.hpp>

__managed__ vector<bucket_t>* table_ptr;
__managed__ int table_size = 1024;

struct table_helper {
 table_helper() {
	CUDA_MALLOC(table_ptr,1);
	new (table_ptr) vector<bucket_t>();
 }
};

vector<bucket_t>& group_table() {
	static table_helper init;
	return *table_ptr;
}

int& group_table_size() {
	return table_size;
}


