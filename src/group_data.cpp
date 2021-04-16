#include <cosmictiger/groups.hpp>
#include <cosmictiger/particle.hpp>
#include <cosmictiger/vector.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/timer.hpp>

#define MIN_GROUP_SIZE 10

static int table_size = 1024;

using bucket_t = vector<group_info_t>;

static vector<bucket_t> table;
static int table_entry_count;

void group_data_rehash() {
}

void group_data_destroy() {
	table = decltype(table)();
}

void group_data_create(particle_set& parts) {
	timer tm;
	tm.start();

	const int nthreads = 2 * hpx::thread::hardware_concurrency();
	std::vector<hpx::future<void>> futs;
	const size_t parts_per_thread = parts.size() / nthreads;
	size_t ngroups;
	ngroups = 0;
	table.resize(table_size);
	group_t last1, last2;
	last1 = last2 = NO_GROUP;
	for (int j = 0; j < parts.size(); j++) {
		const auto id = parts.group(j);
		if (id != NO_GROUP && id != last1 && id != last2) {
			bool rehash = false;
			int index = id % table_size;
			bool found = false;
			auto& entries = table[index];
			for (int k = 0; k < entries.size(); k++) {
				if (entries[k].id == id) {
					found = true;
					break;
				}
			}
			if (!found) {
				group_info_t info;
				*info.count = 0;
				info.ekin = 0.0;
				info.epot = 0.0;
				for (int dim = 0; dim < NDIM; dim++) {
					info.pos[dim] = 0.0;
					info.vel[dim] = 0.0;
				}
				info.ravg = 0.0;
				info.rmax = 0.0;
				info.id = id;
				entries.push_back(info);
				ngroups++;
				if (ngroups == table_size * 2) {
					printf("Rehashing to table_size %i\n", 2 * table_size);
					decltype(table) old_table;
					old_table.swap(table);
					table_size *= 2;
					table.resize(table_size);
					for (int i = 0; i < old_table.size(); i++) {
						auto& entries = old_table[i];
						for (int j = 0; j < entries.size(); j++) {
							int index = entries[j].id % table_size;
							table[index].push_back(entries[j]);
						}
					}
				}
			}
		}
		last2 = last1;
		last1 = id;
	}
	for (int i = 0; i < parts.size(); i += parts_per_thread) {
		const auto func = [i,&parts,parts_per_thread]() {
			const auto jend = std::min(i + parts_per_thread,parts.size());
			for( int j = i; j < jend; j++) {
				const auto id = parts.group(j);
				if( id != NO_GROUP) {
					int index = id % table_size;
					auto& entries = table[index];
					for( int k = 0; k < entries.size(); k++) {
						if( entries[k].id == id) {
							(*entries[k].count)++;
							break;
						}
					}
				}
			}
		};
		futs.push_back(hpx::async(func));
	}
	hpx::wait_all(futs.begin(), futs.end());
	futs.resize(0);
	for (int i = 0; i < table.size(); i++) {
		auto& entries = table[i];
		int j = 0;
		while (j < entries.size()) {
			if (*entries[j].count < MIN_GROUP_SIZE) {
				entries[j] = std::move(entries.back());
				entries.pop_back();
				ngroups--;
			} else {
				j++;
			}
		}
	}
	tm.stop();
	printf("Took %e to create group data found %li groups\n", tm.read(), (size_t) ngroups);
}

void group_data_output(FILE* fp) {
}

void group_data_reduce() {
}
