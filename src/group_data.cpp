#include <cosmictiger/groups.hpp>
#include <cosmictiger/particle.hpp>
#include <cosmictiger/vector.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/timer.hpp>

#define MIN_GROUP_SIZE 10

using group_entry_t = std::pair<group_t,group_info_t>;

static int table_size = 1;
static vector<vector<group_entry_t>> table;
static mutex_type mutex;
static int table_entry_count;

void group_data_rehash() {
	auto old_table = std::move(table);
	table_size *= 2;
	table.resize(table_size);
	for( int i = 0; i < old_table.size(); i++) {
		for( int j = 0; j < old_table[i].size(); j++) {
			int index1 = old_table[i][j].first % table_size;
			table[index1].push_back(old_table[i][j]);
		}
	}
}

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

void group_data_create(particle_set& parts) {
	timer tm;
	tm.start();
	static std::atomic<int> ngroups;
	static std::atomic<int> next_id;
	next_id = 0;
	const auto init_table = [&]() {
		table = decltype(table)();
		table_entry_count = 0;
		table.resize(table_size);
		ngroups = 0;
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
					ngroups++;
					group_entry_t entry;
					entry.first = id;
					index2 = table[index1].size();
					table[index1].push_back(entry);
					table_entry_count++;
				}
				table[index1][index2].second.count++;
				if( table_entry_count > table.size()) {
					printf( "Rehashing\n");
					group_data_rehash();
				}
			}
		}};

	init_table();

	int nthreads = 2 * hpx::thread::hardware_concurrency();
	std::vector<hpx::future<void>> futs(nthreads);
	for (int this_thread = 0; this_thread < nthreads; this_thread++) {
		const auto func = [this_thread,nthreads]() {
			for (int i = this_thread; i < table.size(); i += nthreads) {
				int j = 0;
				while (j < table[i].size()) {
					if (table[i][j].second.count < MIN_GROUP_SIZE) {
						table[i][j] = table[i].back();
						table[i].pop_back();
						ngroups--;
					} else {
						table[i][j].second.next_id = next_id++;
						j++;
					}
				}
			}};
		futs[this_thread] = hpx::async(func);
	}
	hpx::wait_all(futs.begin(), futs.end());
	for (int this_thread = 0; this_thread < nthreads; this_thread++) {
		const auto func = [this_thread,nthreads,&parts]() {
			size_t start = this_thread*parts.size() / nthreads;
			size_t stop = (this_thread+1)*parts.size()/nthreads;
			for (size_t i = start; i < stop; i ++) {
				const auto id = parts.group(i);
				const int index1 = id % table_size;
				bool found = false;
				for( int j = 0; j < table[index1].size(); j++) {
					if( table[index1][j].first == id ) {
						parts.group(i) = table[index1][j].second.next_id;
						found = true;
						break;
					}
				}
				if( !found ) {
					parts.group(i) = NO_GROUP;
				}
			}
		};
		futs[this_thread] = hpx::async(func);
	}
	hpx::wait_all(futs.begin(), futs.end());

	init_table();

	tm.stop();
	printf("Took %e to create group data %i groups found\n", tm.read(), (int) ngroups);
}

void group_data_output(FILE* fp) {
	for (int index1 = 0; index1 < table.size(); index1++) {
		for (int index2 = 0; index2 < table[index1].size(); index2++) {
		}
	}
}

void group_data_reduce() {
	for (int index1 = 0; index1 < table.size(); index1++) {
		int index2 = 0;
		while (index2 < table[index1].size()) {
			group_info_t& data = table[index1][index2].second;
		}
	}
}
