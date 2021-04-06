#include <cosmictiger/groups.hpp>
#include <cosmictiger/particle.hpp>
#include <cosmictiger/vector.hpp>
#include <cosmictiger/hpx.hpp>

#define MIN_GROUP_SIZE 10

using group_entry_t = std::pair<group_t,group_info_t>;

static int table_size = 1024 * 1024;
static vector<vector<group_entry_t>> table;
static mutex_type mutex;

bool group_data_get_entry(group_t id, group_info_t& entry) {
	const int index = id % table_size;
	bool found = false;
	for (int i = 0; i < table.size(); i++) {
		if (table[index][i].first == id) {
			found = true;
			entry = table[index][i].second;
		}
	}
	return found;
}

void group_data_destroy() {
	table = decltype(table)();
}

void group_data_create(const particle_set& parts) {
	table.resize(table_size);
	for (size_t i = 0; i < parts.size(); i++) {
		const group_t id = parts.group(i);
		if (id != NO_GROUP) {
			const int index1 = id % table_size;
			int index2;
			bool found = false;
			for (int i = 0; i < table[index1].size(); i++) {
				if (table[index1][i].first == id) {
					index2 = i;
					found = true;
				}
			}
			if (!found) {
				group_entry_t entry;
				entry.first = id;
				index2 = table[index1].size();
				table[index1].push_back(entry);
			}
			group_info_t& data = table[index1][index2].second;
			const auto vel = parts.vel(i);
			for (int dim = 0; dim < NDIM; dim++) {
				auto x = parts.pos(dim, i).to_double();
				if( x > 0.6 ) {
					printf( "%e\n", x);
					abort();
				}
				if (data.count != 0 && std::abs(x - data.pos[dim]/data.count) > 0.5f) {
					if (x > 0.5) {
				//		x -= 1.0;
					} else {
				//		x += 1.0;
					}
				}
				data.pos[dim] += x;
				data.vel[dim] += vel.a[dim];
				data.ekin += sqr(vel.a[dim]) * 0.5f;
			}
			data.count++;
		}
	}
}

void group_data_output(FILE* fp) {
	for (int index1 = 0; index1 < table.size(); index1++) {
		for (int index2 = 0; index2 < table[index1].size(); index2++) {
			group_t& id = table[index1][index2].first;
			group_info_t& data = table[index1][index2].second;
			fprintf(fp, "%12lli %5i %e %e %e %e\n", id, data.count, data.pos[0], data.pos[1], data.pos[2], data.vtot);
		}
	}
}

void group_data_reduce() {
	for (int index1 = 0; index1 < table.size(); index1++) {
		int index2 = 0;
		while (index2 < table[index1].size()) {
			group_info_t& data = table[index1][index2].second;
			if (data.count < MIN_GROUP_SIZE) {
				table[index1][index2] = table[index1].back();
				table[index1].pop_back();
			} else {
				data.vtot = 0.0;
				for (int dim = 0; dim < NDIM; dim++) {
					data.pos[dim] /= data.count;
					data.vel[dim] /= data.count;
					data.vtot += sqr(data.vel[dim]);
			/*		while(data.pos[dim] > 1.0) {
						data.pos[dim] -= 1.0;
					}
					while(data.pos[dim] < 0.0) {
						data.pos[dim] += 1.0;
					}*/
				}
				data.vtot = std::sqrt(data.vtot);
				index2++;
			}
		}
	}
}
