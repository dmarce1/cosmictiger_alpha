#include <cosmictiger/groups.hpp>
#include <cosmictiger/particle.hpp>
#include <cosmictiger/vector.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/timer.hpp>

#include <unordered_map>

#define MIN_GROUP_SIZE 10

using spinlock_type = hpx::lcos::local::spinlock;
static spinlock_type table_cpu_mutex;
static std::unordered_map<group_t, float> table_cpu_phis;

void cpu_groups_kick_update(group_t id, float phi) {
	if (id != NO_GROUP) {
		std::lock_guard<spinlock_type> lock(table_cpu_mutex);
		if (table_cpu_phis.find(id) == table_cpu_phis.end()) {
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

hpx::future<void> group_data_create(particle_set& parts) {
	timer tm;
	tm.start();
	auto& table = group_table();
	auto& table_size = group_table_size();
	const int nthreads = 2 * hpx::thread::hardware_concurrency();
	std::vector<hpx::future<void>> futs;
	const size_t parts_per_thread = parts.size() / nthreads;
	size_t ngroups;
	ngroups = 0;
	table.resize(table_size);
	group_t last1, last2;
	last1 = last2 = NO_GROUP;
	static timer tm1, tm2, tm3, tm4, tm5;
	tm1.reset();
	tm2.reset();
	tm3.reset();
	tm4.reset();
	tm5.reset();
	tm1.start();
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
	tm1.stop();
	tm2.start();
	for (int i = 0; i < parts.size(); i += parts_per_thread) {
		const auto func = [i,&parts,parts_per_thread,table_size, &table]() {
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
	tm2.stop();
	tm3.start();
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
	tm3.stop();
	tm4.start();
	futs.resize(0);
	for (int i = 0; i < parts.size(); i += parts_per_thread) {
		const auto func = [i,&parts,parts_per_thread,table_size,&table]() {
			const auto jend = std::min(i + parts_per_thread,parts.size());
			for( int j = i; j < jend; j++) {
				auto& id = parts.group(j);
				if( id != NO_GROUP) {
					bool found = false;
					int index = id % table_size;
					auto& entries = table[index];
					for( int k = 0; k < entries.size(); k++) {
						auto& entry = entries[k];
						if( entry.id == id) {
							found = true;
							const auto constrain_range = [](double& r) {
								if( r > 0.5 ) {
									r -= 1.0;
								} else if(r <= -0.5 ) {
									r += 1.0;
								}
							};
							const auto lastid = parts.last_group(j);
							std::lock_guard<group_info_t> lock(entry);
							auto dx = parts.pos(0,j).to_double() - entry.pos[0];
							auto dy = parts.pos(1,j).to_double() - entry.pos[1];
							auto dz = parts.pos(2,j).to_double() - entry.pos[2];
							constrain_range(dx);
							constrain_range(dy);
							constrain_range(dz);
							for( int dim = 0; dim < NDIM; dim++) {
								const float dv = parts.vel(dim,j) - entry.vel[dim];
								entry.vdisp[dim] += dv * dv;
							}
							const auto r = sqrt(fmaf(dx,dx,fmaf(dy,dy,sqr(dz))));

							entry.ravg += r;
							entry.rmax = std::max(entry.rmax,r);
							entry.radii.push_back(r);
							if( lastid != NO_GROUP) {
								if(entry.parents.find(lastid) == entry.parents.end()) {
									entry.parents.insert(std::make_pair(lastid,std::make_shared<int>(0)));
								}
								(*(entry.parents[lastid]))++;
							}
						}
					}
					if( !found) {
						id = NO_GROUP;
					}
				}
			}
		};
		futs.push_back(hpx::async(func));
	}
	hpx::wait_all(futs.begin(), futs.end());
	tm4.stop();
	auto fut =
			hpx::async(
					[nthreads,table,ngroups]() {
						timer tm5;
						tm5.start();
						std::vector<hpx::future<void>> futs;
						for (int tid = 0; tid < nthreads; tid++) {
							const auto func = [tid,nthreads]() {
								auto& table = group_table();
								for( int i = tid; i < table.size(); i+= nthreads) {
									for( int j = 0; j < table[i].size(); j++) {
										auto& entry = table[i][j];
										std::sort(entry.radii.begin(),entry.radii.end());
										const auto N = entry.radii.size();
										for( int dim = 0; dim < NDIM; dim++) {
											entry.vdisp[dim] = std::sqrt(entry.vdisp[dim]/ entry.count);
										}
										if( N % 2 == 0) {
											entry.reff = (entry.radii[N/2-1] + entry.radii[N/2])*0.5;
										} else {
											entry.reff = entry.radii[N/2];
										}
										entry.ravg /= entry.count;
										entry.radii = std::vector<float>();
									}
								}
							};
							futs.push_back(hpx::async(func));
						}
						hpx::wait_all(futs.begin(), futs.end());
						tm5.stop();
						printf("Took %e %e %e %e %e to create group data found %li groups\n", tm1.read(), tm2.read(),tm3.read(),tm4.read(),tm5.read(), (size_t) ngroups);
					});
	return fut;
}

void group_data_save(double scale, int filenum) {
	printf("Saving group data for timestep %i.\n", filenum);
	static auto& table = group_table();
	auto& table_size = group_table_size();
	int max_size = 0;
	int numgroups = 0;
	float avg_size = 0.0;
	float avg_vdisp = 0.0;
	float max_vdisp = 0.0;
	float max_reff = 0.0;
	float avg_reff = 0.0;
	float avg_ekin = 0.0;
	float avg_epot = 0.0;
	float avg_npar = 0.0;
	int max_npar = 0;
	std::vector<int> npars;
	std::vector<float> reffs;
	std::vector<float> vdisps;
	std::vector<int> sizes;
	std::string filename = std::string("groups.") + std::to_string(filenum) + std::string(".dat");
	FILE* fp = fopen(filename.c_str(), "wb");
	if (fp == NULL) {
		printf("Unable to open %s for writing!\n", filename.c_str());
		abort();
	}
	size_t numparts = 0;
	const auto ainv = 1.0 / scale;
	const auto ainv2 = ainv * ainv;
	const auto reff_factor = std::pow(2, 1.0 / 3.0);
	for (int i = 0; i < table.size(); i++) {
		for (int j = 0; j < table[i].size(); j++) {
			auto& entry = table[i][j];
			auto iter = table_cpu_phis.find(entry.id);
			if (iter != table_cpu_phis.end()) {
				entry.epot += iter->second;
			}
			entry.epot *= 0.5;
			entry.ekin *= ainv;
			avg_ekin += entry.ekin;
			avg_epot += entry.epot;
			numparts += entry.count;
			const auto countinv = 1.0 / entry.count;
			entry.epot *= countinv;
			entry.ekin *= countinv;
			entry.reff *= reff_factor;
			for (int dim = 0; dim < NDIM; dim++) {
				while (entry.pos[dim] >= 1.0) {
					entry.pos[dim] -= 1.0;
				}
				while (entry.pos[dim] < 0.0) {
					entry.pos[dim] += 1.0;
				}
			}
			numgroups++;
			float vdx = entry.vdisp[0];
			float vdy = entry.vdisp[1];
			float vdz = entry.vdisp[2];
			float vdisp = sqrt(fmaf(vdx, vdx, fmaf(vdy, vdy, sqr(vdz))));
			max_size = std::max(max_size, entry.count);
			max_reff = std::max(max_reff, entry.reff);
			max_vdisp = std::max(vdisp, max_vdisp);
			avg_size += entry.count;
			avg_reff += entry.reff;
			avg_vdisp += vdisp;
			npars.push_back(entry.parents.size());
			reffs.push_back(entry.reff);
			vdisps.push_back(vdisp);
			sizes.push_back(entry.count);
			avg_npar += entry.parents.size();
			max_npar = std::max(max_npar, (int) entry.parents.size());
			fwrite(&entry.id, sizeof(entry.id), 1, fp);
			fwrite(&entry.count, sizeof(entry.count), 1, fp);
			fwrite(&entry.pos, sizeof(entry.pos), 1, fp);
			fwrite(&entry.vel, sizeof(entry.vel), 1, fp);
			fwrite(&entry.vdisp, sizeof(entry.vdisp), 1, fp);
			fwrite(&entry.epot, sizeof(float), 1, fp);
			fwrite(&entry.ekin, sizeof(float), 1, fp);
			fwrite(&entry.rmax, sizeof(float), 1, fp);
			fwrite(&entry.ravg, sizeof(float), 1, fp);
			fwrite(&entry.reff, sizeof(float), 1, fp);
			int num_parents = entry.parents.size();
			fwrite(&num_parents, sizeof(int), 1, fp);
			for (auto i = entry.parents.begin(); i != entry.parents.end(); i++) {
				fwrite(&i->first, sizeof(i->first), 1, fp);
				fwrite(&i->second, sizeof(i->second), 1, fp);
			}
		}
	}
	fclose(fp);
	if (numgroups) {
		avg_size /= numgroups;
		avg_reff /= numgroups;
		avg_vdisp /= numgroups;
		avg_ekin /= numparts;
		avg_epot /= numparts;
		avg_npar /= numgroups;
		auto fut1 = hpx::async([&]() {std::sort(npars.begin(), npars.end());});
		auto fut2 = hpx::async([&]() {std::sort(reffs.begin(), reffs.end());});
		auto fut3 = hpx::async([&]() {std::sort(vdisps.begin(), vdisps.end());});
		auto fut4 = hpx::async([&]() {std::sort(sizes.begin(), sizes.end());});
		fut1.get();
		fut2.get();
		fut3.get();
		fut4.get();
		int med_npar = npars[numgroups / 2];
		float med_reff = reffs[numgroups / 2];
		float med_vdisp = vdisps[numgroups / 2];
		int med_size = sizes[numgroups / 2];
		printf("Group Statistics for %i groups\n", numgroups);
		printf("\t%.3f%% of particles are in a group\n", 100.0 * numparts / global().opts.nparts);
		printf("\tmaximum size  : %i particles\n", max_size);
		printf("\taverage size  : %f particles\n", avg_size);
		printf("\tmedian  size  : %i particles\n", med_size);
		printf("\tmaximum reff   : %e\n", max_reff);
		printf("\taverage reff   : %e\n", avg_reff);
		printf("\tmedian  reff   : %e\n", med_reff);
		printf("\tmaximum velocity dispersion : %e\n", max_vdisp);
		printf("\taverage velocity dispersion : %e\n", avg_vdisp);
		printf("\tmedian  velocity dispersion : %e\n", med_vdisp);
		printf("\taverage kinetic energy   : %e\n", avg_ekin);
		printf("\taverage potential energy : %e\n", avg_epot);
		printf("\tmaximum number of parents : %i\n", max_npar);
		printf("\taverage number of parents : %f\n", avg_npar);
		printf("\tmedian  number of parents : %i\n", med_npar);
	}
}

