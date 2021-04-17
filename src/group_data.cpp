#include <cosmictiger/groups.hpp>
#include <cosmictiger/particle.hpp>
#include <cosmictiger/vector.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/timer.hpp>

#include <unordered_map>

#define MIN_GROUP_SIZE 10

using spinlock_type = hpx::lcos::local::spinlock;
static spinlock_type table_cpu_mutex;
static std::unordered_map<group_t,float> table_cpu_phis;

void cpu_groups_kick_update(group_t id, float phi) {
	if (id != NO_GROUP) {
		std::lock_guard<spinlock_type> lock(table_cpu_mutex);
		if( table_cpu_phis.find(id) == table_cpu_phis.end()) {
			table_cpu_phis[id] = 0.0;
		}
		table_cpu_phis[id] += phi;
	}
}

void group_data_destroy() {
	auto& table = group_table();
	table = vector<bucket_t>();
	table_cpu_phis = decltype(table_cpu_phis)();
}

void group_data_create(particle_set& parts) {
	timer tm;
	tm.start();
	static auto& table = group_table();
	auto& table_size = group_table_size();
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
				info.count = 0;
				info.ekin = 0.0;
				info.epot = 0.0;
				for (int dim = 0; dim < NDIM; dim++) {
					info.vdisp[dim] = 0.0;
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
					vector<bucket_t> old_table;
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
		const auto func = [i,&parts,parts_per_thread,table_size]() {
			const auto jend = std::min(i + parts_per_thread,parts.size());
			for( int j = i; j < jend; j++) {
				const auto id = parts.group(j);
				if( id != NO_GROUP) {
					int index = id % table_size;
					auto& entries = table[index];
					for( int k = 0; k < entries.size(); k++) {
						auto& entry = entries[k];
						if( entry.id == id) {
							std::lock_guard<group_info_t> lock(entry);
							array<fixed32,NDIM> pos;
							for( int dim = 0; dim < NDIM; dim++) {
								double pos = parts.pos(dim,j).to_double();
								if( entry.count) {
									const auto diff = pos - entry.pos[dim] / entry.count;
									if( diff > 0.5 ) {
										pos -= 1.0;
									} else if( diff < -0.5) {
										pos += 1.0;
									}
									entry.pos[dim] += pos;
								}
								const float vx = parts.vel(0,j);
								const float vy = parts.vel(1,j);
								const float vz = parts.vel(2,j);
								entry.ekin += 0.5f * fmaf(vx,vx,fmaf(vy,vy,sqr(vz)));
								entry.vel[0] += vx;
								entry.vel[1] += vy;
								entry.vel[2] += vz;
							}
							entry.count++;
							break;
						}
					}
				}
			}
		};
		futs.push_back(hpx::async(func));
	}
	hpx::wait_all(futs.begin(), futs.end());
	for (int i = 0; i < table.size(); i++) {
		auto& entries = table[i];
		int j = 0;
		while (j < entries.size()) {
			auto& entry = entries[j];
			if (entry.count < MIN_GROUP_SIZE) {
				entry = std::move(entries.back());
				entries.pop_back();
				ngroups--;
			} else {
				for (int dim = 0; dim < NDIM; dim++) {
					entry.pos[dim] /= entry.count;
					entry.vel[dim] /= entry.count;
				}
				j++;
			}
		}
	}
	futs.resize(0);
	for (int i = 0; i < parts.size(); i += parts_per_thread) {
		const auto func = [i,&parts,parts_per_thread,table_size]() {
			const auto jend = std::min(i + parts_per_thread,parts.size());
			for( int j = i; j < jend; j++) {
				const auto id = parts.group(j);
				if( id != NO_GROUP) {
					int index = id % table_size;
					auto& entries = table[index];
					for( int k = 0; k < entries.size(); k++) {
						auto& entry = entries[k];
						if( entry.id == id) {
							const auto lastid = parts.last_group(j);
							std::lock_guard<group_info_t> lock(entry);
							const auto dx = parts.pos(0,j).to_double() - entry.pos[0];
							const auto dy = parts.pos(1,j).to_double() - entry.pos[1];
							const auto dz = parts.pos(2,j).to_double() - entry.pos[2];
							for( int dim = 0; dim < NDIM; dim++) {
								const float dv = parts.vel(dim,j) - entry.vel[dim];
								entry.vdisp[dim] += dv * dv;
							}
							const auto r = sqrt(fmaf(dx,dx,fmaf(dy,dy,sqr(dz))));

							entry.ravg += r;
							entry.rmax = std::max(entry.rmax,r);
							entry.radii.push_back(r);
							if(entry.parents.find(lastid) == entry.parents.end()) {
								entry.parents.insert(std::make_pair(lastid,std::make_shared<int>(0)));
							}
							(*(entry.parents[lastid]))++;
						}
					}
				}
			}
		};
		futs.push_back(hpx::async(func));
	}
	hpx::wait_all(futs.begin(), futs.end());
	futs.resize(0);
	for (int tid = 0; tid < nthreads; tid++) {
		const auto func = [tid,nthreads]() {
			for( int i = tid; i < table.size(); i+= nthreads) {
				for( int j = 0; j < table[i].size(); j++) {
					auto& entry = table[i][j];
					std::sort(entry.radii.begin(),entry.radii.end());
					const auto N = entry.radii.size();
					for( int dim = 0; dim < NDIM; dim++) {
						entry.vdisp[dim] = std::sqrt(entry.vdisp[dim]) / entry.count;
					}
					if( N % 2 == 0) {
						entry.r50 = (entry.radii[N/2-1] + entry.radii[N/2])*0.5;
					} else {
						entry.r50 = entry.radii[N/2];
					}
					entry.radii = std::vector<float>();
				}
			}
		};
		futs.push_back(hpx::async(func));
	}
	hpx::wait_all(futs.begin(), futs.end());
	tm.stop();
	printf("Took %e to create group data found %li groups\n", tm.read(), (size_t) ngroups);
}

void group_data_output(FILE* fp) {
}

void group_data_reduce() {
}
