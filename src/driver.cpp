#include <cosmictiger/driver.hpp>
#include <cosmictiger/kick_return.hpp>
#include <cosmictiger/drift.hpp>
#include <cosmictiger/time.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/cosmos.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/map.hpp>
#include <cosmictiger/initial.hpp>
#include <cosmictiger/groups.hpp>
#include <cosmictiger/power.hpp>
#include <cosmictiger/driver.hpp>
#include <cosmictiger/particle_server.hpp>

double T0;
#define NTIMESTEP 100.0

tree_ptr build_tree(int min_rung, double theta, size_t& num_active, tree_stats& stats, double& tm, bool group_sort =
		false) {
	timer time;
	time.start();
	static bool last_was_group_sort = true;
	static int last_bucket_size = global().opts.bucket_size;
	int bucket_size = global().opts.bucket_size;
	if ((last_was_group_sort || bucket_size != last_bucket_size) && !group_sort) {
		tree_data_initialize(TREE_KICK);
		last_was_group_sort = false;
	} else if (group_sort) {
		last_was_group_sort = true;
		tree_data_initialize(TREE_GROUPS);
	}
	last_bucket_size = bucket_size;
	static particle_set *parts_ptr = nullptr;
	tree root;
	tree_ptr root_ptr;

	sort_params params;
	params.min_rung = min_rung;
	params.theta = theta;
	params.group_sort = group_sort;
	particle_server pserv;
	pserv.apply_domain_decomp();
	sort_return rc = root.sort(params);
	root_ptr = rc.check;
	num_active = rc.active_parts;
	time.stop();
	tm = time.read();
	stats = rc.stats;
	return root_ptr;

}

int kick(tree_ptr root_ptr, double theta, double a, int min_rung, bool full_eval, bool first_call, bool groups,
		double& tm) {
	timer time;
	time.start();
	kick_return_init(min_rung);
	for (int pass = 0; pass < 2; pass++) {
		kick_params_type *params_ptr;
		CUDA_MALLOC(params_ptr, 1);
		new (params_ptr) kick_params_type();
		params_ptr->dry_run = (pass == 0);
		params_ptr->tptr = root_ptr;
		params_ptr->dchecks.push(root_ptr);
		params_ptr->echecks.push(root_ptr);
		params_ptr->rung = min_rung;
		params_ptr->full_eval = full_eval;
		params_ptr->theta = theta;
		params_ptr->scale = a;
		params_ptr->groups = groups;
		array<fixed32, NDIM> Lpos;
		expansion<float> L;
		for (int i = 0; i < LP; i++) {
			L[i] = 0.f;
		}
		for (int dim = 0; dim < NDIM; dim++) {
			Lpos[dim] = 0.5;
		}
		params_ptr->L[0] = L;
		params_ptr->Lpos[0] = Lpos;
		params_ptr->first = first_call;
		params_ptr->t0 = T0;
		params_ptr->scale = a;
		tree::kick(params_ptr);
		first_call = false;
		time.stop();
		tm = time.read();
		params_ptr->kick_params_type::~kick_params_type();
		CUDA_FREE(params_ptr);
	}
	if (full_eval) {
		kick_return_show();
	}
	tree::cleanup();
	return kick_return_max_rung();
}

int drift(particle_set& parts, double dt, double a0, double a1, double*ekin, double*momx, double*momy, double*momz,
		double tau, double tau_max, double& tm) {
	timer time;
	time.start();
	drift_return rc = drift(dt, a0, tau, tau_max);
	*ekin = rc.ekin;
	*momx = rc.momx;
	*momy = rc.momy;
	*momz = rc.momz;
	time.stop();
	tm = time.read();
	return rc.map_cnt;
}

#define EVAL_FREQ 25

void save_to_file(int step, time_type itime, double time, double a, double cosmicK, double esum0);
void load_from_file_remote();

HPX_PLAIN_ACTION(save_to_file);
HPX_PLAIN_ACTION(load_from_file_remote);

void save_to_file(int step, time_type itime, double time, double a, double cosmicK, double esum0) {
	particle_server pserv;
	auto& parts = pserv.get_particle_set();
	std::vector<hpx::future<void>> futs;
	if (hpx_rank() == 0) {
		for (int i = 1; i < hpx_size(); i++) {
			futs.push_back(hpx::async<save_to_file_action>(hpx_localities()[i], step, itime, time, a, cosmicK, esum0));
		}
	}

	std::string filename = std::string("checkpoint.") + std::to_string(step) + std::string(".dat");
	if (hpx_size() > 1) {
		filename += std::string(".") + std::to_string(hpx_rank());
	}
	PRINT("Saving %s...", filename.c_str());
	fflush(stdout);
	FILE* fp = fopen(filename.c_str(), "wb");
	if (!fp) {
		PRINT("Unable to open %s\n", filename.c_str());
		abort();
	}
	fwrite(&step, sizeof(int), 1, fp);
	fwrite(&itime, sizeof(time_type), 1, fp);
	fwrite(&time, sizeof(double), 1, fp);
	fwrite(&a, sizeof(double), 1, fp);
	fwrite(&cosmicK, sizeof(double), 1, fp);
	fwrite(&esum0, sizeof(double), 1, fp);
	parts.save_to_file(fp);
	maps_to_file(fp);
	fclose(fp);
	PRINT(" done\n");

	hpx::wait_all(futs.begin(), futs.end());
}

void load_from_file_remote() {
	particle_server pserv;
	auto& parts = pserv.get_particle_set();
	std::vector<hpx::future<void>> futs;
	if (hpx_rank() == 0) {
		for (int i = 1; i < hpx_size(); i++) {
			futs.push_back(hpx::async<load_from_file_remote_action>(hpx_localities()[i]));
		}
	}
	std::string filename = global().opts.checkpt_file;
	int step;
	time_type itime;
	double time;
	double a;
	double cosmicK;
	double esum0;
	if (hpx_size() > 1) {
		filename += std::string(".") + std::to_string(hpx_rank());
	}
	FILE* fp = fopen(filename.c_str(), "rb");
	if (!fp) {
		PRINT("Unable to open %s\n", filename.c_str());
		abort();
	}
	FREAD(&step, sizeof(int), 1, fp);
	FREAD(&itime, sizeof(time_type), 1, fp);
	FREAD(&time, sizeof(double), 1, fp);
	FREAD(&a, sizeof(double), 1, fp);
	FREAD(&cosmicK, sizeof(double), 1, fp);
	FREAD(&esum0, sizeof(double), 1, fp);
	parts.load_from_file(fp);
	maps_from_file(fp);
	fclose(fp);
	hpx::wait_all(futs.begin(), futs.end());

}

void load_from_file(int& step, time_type& itime, double& time, double& a, double& cosmicK, double& esum0) {
	particle_server pserv;
	auto& parts = pserv.get_particle_set();
	std::string filename = global().opts.checkpt_file;
	PRINT("Loading %s...", filename.c_str());
	if (hpx_size() > 1) {
		filename += std::string(".") + std::to_string(hpx_rank());
	}
	FILE* fp = fopen(filename.c_str(), "rb");
	if (!fp) {
		PRINT("Unable to open %s\n", filename.c_str());
		abort();
	}
	FREAD(&step, sizeof(int), 1, fp);
	FREAD(&itime, sizeof(time_type), 1, fp);
	FREAD(&time, sizeof(double), 1, fp);
	FREAD(&a, sizeof(double), 1, fp);
	FREAD(&cosmicK, sizeof(double), 1, fp);
	FREAD(&esum0, sizeof(double), 1, fp);
	parts.load_from_file(fp);
	maps_from_file(fp);
	fclose(fp);
	load_from_file_remote();
	PRINT(" done\n");

}

std::pair<int, hpx::future<void>> find_groups(particle_set& parts, double& time) {
	timer tm;
	tm.start();
	double sort_tm;
	size_t num_active;
	tree_stats stats;
	PRINT("Finding Groups\n");

	tree_ptr root_ptr = build_tree(0, 1.0, num_active, stats, sort_tm, true);

	unified_allocator alloc;
	alloc.reset();

	group_param_type *params_ptr;
	CUDA_MALLOC(params_ptr, 1);
	new (params_ptr) group_param_type();
	params_ptr->self = root_ptr;
	params_ptr->link_len = 1.0 / pow(global().opts.nparts, 1.0 / 3.0) * 0.2;
	params_ptr->parts = parts.get_virtual_particle_set();
	params_ptr->checks.push(root_ptr);
	params_ptr->first_round = true;
	int iters = 1;
	parts.init_groups();
	tree_database_set_groups();
	size_t active = find_groups(params_ptr).get();
	PRINT("%li nodes active\n", active);
	while (active) {
		tree_database_set_groups();
		params_ptr->self = root_ptr;
		params_ptr->checks.resize(0);
		params_ptr->checks.push(root_ptr);
		params_ptr->first_round = false;
		iters++;
		active = find_groups(params_ptr).get();
		PRINT("%li nodes active\n", active);
	}
	params_ptr->~group_param_type();
	tree_data_free_all();
	CUDA_FREE(params_ptr);
	alloc.reset();
	auto fut = group_data_create(parts);
	tm.stop();
	time = tm.read();
	tree::cleanup();
	return std::make_pair(iters, std::move(fut));
}

void drive_cosmos() {

	bool have_checkpoint = global().opts.checkpt_file != "";

	int max_iter = 100;

	int iter = 0;
	int max_rung = 0;
	time_type itime = 0;
	double kick_tm, drift_tm, sort_tm;
	double kick_total, drift_total, sort_total;
	kick_total = drift_total = sort_total = 0.0;
	double a;
	double Ka;
	double z;
	double cosmicK = 0.0;
	double theta;
	double pot;
	double esum0;
	double parts_total;
	double time_total;
	double time;

	particle_server pserv;
	pserv.init();
	auto& parts = pserv.get_particle_set();

	if (!have_checkpoint) {
		//pserv.load_NGenIC();
		initial_conditions(parts);
		itime = 0;
		iter = 0;
		z = global().opts.z0;
		a = 1.0 / (z + 1.0);
		time = 0.0;
	} else {
		load_from_file(iter, itime, time, a, cosmicK, esum0);
		z = 1.0 / a - 1.0;
	}
	parts_total = 0.0;
	time_total = 0.0;
	T0 = cosmos_age(1.0 / (1.0 + global().opts.z0)) / NTIMESTEP;
	timer tm;
	tm.start();
	double kin, momx, momy, momz;
	double partfac = 1.0 / global().opts.nparts;
	if (!have_checkpoint) {
		drift(parts, 0.0, a, a, &kin, &momx, &momy, &momz, 0.0, NTIMESTEP * T0, drift_tm);
		PRINT("Starting ekin = %e\n", a * kin * partfac);
	}
	timer checkpt_tm;
	checkpt_tm.start();
	int silo_outs = 0;
	double real_time = 0.0;
	double tree_use;
	hpx::future<void> group_fut;
	bool first_step_from_restart = true;
	do {
		checkpt_tm.stop();
		if (checkpt_tm.read() > global().opts.checkpt_freq) {
			checkpt_tm.reset();
			checkpt_tm.start();
			save_to_file(iter, itime, time, a, cosmicK, esum0);
		} else {
			checkpt_tm.start();
		}
		if (iter % 10 == 0) {
			PRINT(
					"%9s %4s %9s %4s %4s %4s %9s %9s %9s %9s %4s %4s %9s %9s %9s %9s %9s %9s %9s %9s %9s %9s %9s %9s %9s %9s\n",
					"GB", "iter", "mapped", "maxd", "mind", "ed", "avgd", "ppl", "actv", "arena", "min", "max", "time", "dt",
					"theta", "a", "z", "pot", "kin", "cosmicK", "esum", "sort", "kick", "drift", "tot", "srate");
		}
		static double last_theta = -1.0;
		auto opts = global().opts;
		int bucket_size = global().opts.bucket_size;
		if( LORDER == 6 ) {
			if (z > 20) {
				theta = 0.4;
			} else if (z > 2) {
				theta = 0.55;
			} else {
				theta = 0.7;
			}
		} else if( LORDER == 7 ) {
			if (z > 20) {
				theta = 0.47;
			} else if (z > 2) {
				theta = 0.60;
			} else {
				theta = 0.74;
			}
		} else if( LORDER == 8 ) {
			if (z > 20) {
				theta = 0.52;
			} else if (z > 2) {
				theta = 0.65;
			} else {
				theta = 0.78;
			}
		}
		opts.bucket_size = bucket_size;
		global_set_options(opts);
		if (theta != last_theta) {
			reset_list_sizes();
			last_theta = theta;
		}
		const auto min_r = min_rung(itime);
		size_t num_active;
		tree_stats stats;
		const bool full_eval = min_r <= 0 || first_step_from_restart;
		//	const bool full_eval = false;
		double group_tm;
		bool groups = z < 20.0 && global().opts.groups;
		bool power = global().opts.power;
		unified_allocator alloc;
	//	tree_data_free_all();
	//	tree_data_initialize(TREE_KICK);
		if (full_eval && (power || groups)) {
			alloc.reset();
			if (power) {
				;
				tree_data_free_all();
				timer tm;
				PRINT("Computing matter power spectrum\n");
				tm.start();
				matter_power_spectrum(time + 0.5);
				tm.stop();
				PRINT("Took %e seconds\n", tm.read());
				if (!groups) {
					tree_data_initialize(TREE_KICK);
				}
			}
			if (groups) {
				ERROR()
				;
				auto rc = find_groups(parts, group_tm);
				PRINT("Finding groups took %e s and %i iterations\n", group_tm, rc.first);
				group_fut = std::move(rc.second);
			}
		}
		tree_ptr root_ptr = build_tree(min_r, theta, num_active, stats, sort_tm);
		tree_use = tree_data_use();
		max_rung = kick(root_ptr, theta, a, min_rung(itime), full_eval, iter == 0, groups && full_eval, kick_tm);
		alloc.reset();
		if (full_eval && groups) {
			group_fut.get();
			group_data_save(a, time + 0.5);
			group_data_destroy();
		}
		const auto silo_int = global().opts.silo_interval;
		if (silo_int > 0) {
			ERROR()
			;
			if (full_eval) {
				std::string filename = "points." + std::to_string(silo_outs) + ".silo";
				silo_outs++;
			}
		}

		kick_return kr = kick_return_get();
		if (full_eval) {
			pot = 0.5 * kr.phis / a;
		}
		double dt = T0 / double(size_t(1) << max_rung);
		time += dt / T0;
		double a0 = a;
		double datau1 = cosmos_dadtau(a);
		double datau2 = cosmos_dadtau(a + datau1 * dt);
		real_time += a * dt * 0.5;
		a += (0.5 * datau1 + 0.5 * datau2) * dt;
		real_time += a * dt * 0.5;
		z = 1.0 / a - 1.0;
		if (global().opts.map_size > 0) {
			load_and_save_maps(time * T0 - dt, NTIMESTEP * T0);
		}
		int mapped_cnt = drift(parts, dt, a0, a, &kin, &momx, &momy, &momz, T0 * time - dt, NTIMESTEP * T0, drift_tm);
		cosmicK += kin * (a - a0);
		double sum = a * (pot + kin) + cosmicK;
		//	PRINT( "%e %e %e %e\n", a, pot, kin, cosmicK);
		if (iter == 0 && !have_checkpoint) {
			esum0 = sum;
		}
		sum = (sum - esum0) / (std::abs(a * kin) + std::abs(a * pot) + std::abs(cosmicK));
		parts_total += num_active;
		double act_pct = 100.0 * (double) num_active * partfac;
		tm.stop();
		time_total += tm.read();
		tm.reset();
		tm.start();
		double science_rate = parts_total / time_total;
		drift_total += drift_tm;
		sort_total += sort_tm;
		kick_total += kick_tm;
		double total_time = drift_tm + sort_tm + kick_tm;
		double parts_per_leaf = (double) parts.size() / stats.nleaves;
		double avg_depth = std::log(stats.nleaves) / std::log(2.0);
//		if (full_eval) {
		double tfac = 1.0 / 365.25 / 60.0 / 60.0 / 24.0 / global().opts.hubble * global().opts.code_to_s;
		double years = real_time * tfac;
		double dtyears = dt * tfac * a;
		PRINT(
				"%9.2e %4i %9i %4i %4i %4i %9.3f %9.2e %9.2f%% %9.2f%% %4i %4i %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n",
				cuda_unified_total() / 1024.0 / 1024 / 1024, iter, mapped_cnt, stats.max_depth, stats.min_depth,
				stats.e_depth, avg_depth, parts_per_leaf, act_pct, tree_use * 100.0, min_r, max_rung, time, dt, theta, a0,
				z, a * pot * partfac, a * kin * partfac, cosmicK * partfac, sum, sort_tm, kick_tm, drift_tm, total_time,
				science_rate);
//		} else {
//			PRINT("%4i %9.2f%% %4i %4i %9.2e %9.2e %9.2e %9.2e %9.2e %9s %9.2e %9.2e %9s %9.2e %9.2e %9.2e %9.2e %9.2e\n",
//					iter, act_pct, min_r, max_rung, time, dt, theta, a0, z, "n/a", a * kin * partfac, cosmicK * partfac,
//					"n/a", sort_tm, kick_tm, drift_tm, total_time, science_rate);
//		}
		itime = inc(itime, max_rung);
		iter++;
		first_step_from_restart = false;
	} while (z > global().opts.z1);
	save_to_file(0, itime, time, a, cosmicK, esum0);

	double total_time = drift_total + sort_total + kick_total;
	PRINT("Sort  time = %e (%.2f) %%\n", sort_total, sort_total / total_time * 100.0);
	PRINT("Kick  time = %e (%.2f) %%\n", kick_total, kick_total / total_time * 100.0);
	PRINT("Drift time = %e (%.2f) %%\n", drift_total, drift_total / total_time * 100.0);
	PRINT("Total time = %e\n", total_time);
}
