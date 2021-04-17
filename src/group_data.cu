#include <cosmictiger/groups.hpp>

__managed__ vector<bucket_t>* table_ptr;
__managed__ int table_size = 1;

struct table_helper {
	table_helper() {
		CUDA_MALLOC(table_ptr, 1);
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

__device__
void gpu_groups_kick_update(group_t id, float phi) {
	if (id != NO_GROUP) {
		const int index1 = id % table_size;
		bool found = false;
		auto& table = *table_ptr;
		int index2;
		for (index2 = 0; index2 < table[index1].size(); index2++) {
			if (table[index1][index2].id == id) {
				found = true;
				break;
			}
		}
		if (found) {
			auto& entry = table[index1][index2];
			atomicAdd(&entry.epot, phi);
		}
	}
}
