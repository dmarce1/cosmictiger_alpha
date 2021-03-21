#include <cosmictiger/driver.hpp>
#include <cosmictiger/kick_return.hpp>
#include <cosmictiger/drift.hpp>
#include <cosmictiger/time.hpp>
#include <cosmictiger/timer.hpp>

double T0;

double da_tau(double a) {
	const auto H = global().opts.H0 * global().opts.hubble;
	const auto omega_m = global().opts.omega_m;
	const auto omega_r = 7e-3;
	const auto omega_lambda = 1.0 - omega_r - omega_m;
	return H * a * a * std::sqrt(omega_m / (a * a * a) + omega_r / (a * a * a * a) + omega_lambda);
}

double uni_age(double a) {
	double dt = 1.0e-6;
	double t = 0.0;
	while (a < 1.0) {
		double datau1 = da_tau(a);
		double datau2 = da_tau(a + datau1 * dt);
		a += (0.5 * datau1 + 0.5 * datau2) * dt;
		t += dt;
	}
	return t;
}

tree build_tree(particle_set& parts, int min_rung, size_t& num_active, double& tm) {
	timer time;
	time.start();
	tree::set_particle_set(&parts);
	particle_set *parts_ptr;
	CUDA_MALLOC(parts_ptr, sizeof(particle_set));
	new (parts_ptr) particle_set(parts.get_virtual_particle_set());
	tree::cuda_set_kick_params(parts_ptr);
	tree root;
	sort_params params;
	params.min_rung = min_rung;
	sort_return rc = root.sort(params);
	num_active = rc.active_parts;
	time.stop();
	tm = time.read();
	return root;

}

int kick(tree root, double theta, double a, int min_rung, bool full_eval, double& tm) {
	timer time;
	time.start();
	static bool first_call = true;
	tree_ptr root_ptr;
	root_ptr.ptr = (uintptr_t) &root;
	kick_params_type *params_ptr;
	CUDA_MALLOC(params_ptr, 1);
	new (params_ptr) kick_params_type();
	params_ptr->dchecks.push(root_ptr);
	params_ptr->echecks.push(root_ptr);
	params_ptr->rung = min_rung;
	params_ptr->full_eval = full_eval;
	params_ptr->theta = theta;
	params_ptr->scale = a;
	array<fixed32, NDIM> Lpos;
	expansion<float> L;
	for (int i = 0; i < LP; i++) {
		L[i] = 0.f;
	}
	for (int dim = 0; dim < NDIM; dim++) {
		Lpos[dim] = 0.5;
	}
	kick_return_init(min_rung);
	params_ptr->L[0] = L;
	params_ptr->Lpos[0] = Lpos;
	params_ptr->first = first_call;
	params_ptr->t0 = T0;
	params_ptr->scale = a;
	root.kick(params_ptr).get();
	tree::cleanup();
	managed_allocator<tree>::cleanup();
	if (full_eval) {
		kick_return_show();
	}
	first_call = false;
	time.stop();
	tm = time.read();
	return kick_return_max_rung();
}

void drift(particle_set& parts, double dt, double a0, double a1, double*ekin, double*momx, double*momy, double*momz,
		double& tm) {
	timer time;
	time.start();
	drift_particles(parts.get_virtual_particle_set(), dt, a0, a1, ekin, momx, momy, momz);
	time.stop();
	tm = time.read();
}

#define EVAL_FREQ 25

void drive_cosmos() {
	particle_set parts(global().opts.nparts);
	parts.load_particles("ics");
//	printf( "Particles loaded\n");
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
	z = global().opts.z0;
	a = 1.0 / (z + 1.0);
	double cosmicK = 0.0;
	double theta;
	double pot;
	double esum0;
	double parts_total;
	double time_total;
	parts_total = 0.0;
	time_total = 0.0;
	T0 = uni_age(a);
	timer tm;
	tm.start();
	do {
		if (iter % 25 == 0) {
			printf("%4s %9s %4s %4s %9s %9s %9s %9s %9s %9s %9s %9s %9s %9s %9s %9s %9s\n", "iter", "actv", "min", "max",
					"time", "dt", "theta", "a", "z", "pot", "kin", "cosmicK", "esum", "sort", "kick", "drift", "srate");
		}
		if (z > 20.0) {
			theta = 0.4;
		} else if (z > 2.0) {
			theta = 0.55;
		} else {
			theta = 0.7;
		}
		double time = double(itime) * T0 / double(std::numeric_limits<time_type>::max());
		const auto min_r = min_rung(itime);
		size_t num_active;
		tree root = build_tree(parts, min_r, num_active, sort_tm);
	//	const bool full_eval = min_r <= 7;
		const bool full_eval = false;
		max_rung = kick(root, theta, a, min_rung(itime), full_eval, kick_tm);
		kick_return kr = kick_return_get();
		if (full_eval) {
			pot = 0.5 * kr.phis / a;
		}
		double dt = T0 / double(size_t(1) << max_rung);
		double a0 = a;
		double datau1 = da_tau(a);
		double datau2 = da_tau(a + datau1 * dt);
		a += (0.5 * datau1 + 0.5 * datau2) * dt;
		z = 1.0 / a - 1.0;
		double kin, momx, momy, momz;
		drift(parts, dt, a0, a, &kin, &momx, &momy, &momz, drift_tm);
		cosmicK += kin * (a - a0);
		double sum = a * (pot + kin) + cosmicK;
		//	printf( "%e %e %e %e\n", a, pot, kin, cosmicK);
		if (iter == 0) {
			esum0 = sum;
		}
		sum = (sum - esum0) / (std::abs(a * kin) + std::abs(a * pot) + std::abs(cosmicK));
		double partfac = 1.0 / global().opts.nparts;
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
		if (full_eval) {
			printf("%4i %9.2f%% %4i %4i %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n",
					iter, act_pct, min_r, max_rung, time, dt, theta, a0, z, a * pot * partfac, a * kin * partfac,
					cosmicK * partfac, sum, sort_tm, kick_tm, drift_tm, science_rate);
		} else {
			printf("%4i %9.2f%% %4i %4i %9.2e %9.2e %9.2e %9.2e %9.2e %9s %9.2e %9.2e %9s %9.2e %9.2e %9.2e %9.2e\n", iter,
					act_pct, min_r, max_rung, time, dt, theta, a0, z, "n/a", a * kin * partfac, cosmicK * partfac, "n/a",
					sort_tm, kick_tm, drift_tm, science_rate);
		}
		itime = inc(itime, max_rung);
		if (iter >= 100) {
	//		break;
		}
		iter++;
	} while (z > 0.0);
	double total_time = drift_total + sort_total + kick_total;
	printf("Sort  time = %e (%.2f) %%\n", sort_total, sort_total / total_time * 100.0);
	printf("Kick  time = %e (%.2f) %%\n", kick_total, kick_total / total_time * 100.0);
	printf("Drift time = %e (%.2f) %%\n", drift_total, drift_total / total_time * 100.0);
}
