#include <cosmictiger/driver.hpp>
#include <cosmictiger/kick_return.hpp>
#include <cosmictiger/drift.hpp>
#include <cosmictiger/time.hpp>
#include <cosmictiger/timer.hpp>

#define T0 1.0

tree build_tree(particle_set& parts, double& tm) {
	timer time;
	time.start();
	tree::set_particle_set(&parts);
	particle_set *parts_ptr;
	CUDA_MALLOC(parts_ptr, sizeof(particle_set));
	new (parts_ptr) particle_set(parts.get_virtual_particle_set());
	tree::cuda_set_kick_params(parts_ptr);
	tree root;
	root.sort();
	time.stop();
	tm = time.read();
	return root;

}

int kick(tree root, int min_rung, double& tm) {
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
	root.kick(params_ptr).get();
	tree::cleanup();
	managed_allocator<tree>::cleanup();
//	kick_return_show();
	first_call = false;
	time.stop();
	tm = time.read();
	return kick_return_max_rung();
}

void drift(particle_set& parts, double dt, double& tm) {
	timer time;
	time.start();
	drift_particles(parts.get_virtual_particle_set(), dt, 1.0, 1.0);
	time.stop();
	tm = time.read();
}

void drive_cosmos() {
	particle_set parts(global().opts.nparts);
	parts.load_particles("ics");
	int max_iter = 100;

	int iter = 0;
	int max_rung = 0;
	time_type itime = 0;
	double tm;
	while (iter < max_iter) {
		double tm = double(itime) * T0 / std::numeric_limits<time_type>::max();
		printf("Time = %e Min Rung = %i Max Rung = %i\n", tm,  min_rung(itime), max_rung);
		tree root = build_tree(parts, tm);
		printf("Building tree took %e s\n", tm);
		max_rung = kick(root, min_rung(itime), tm);
		printf("Kicking took       %e s \n", tm);
		double dt = T0 / double(1 << max_rung);
		drift(parts, dt, tm);
		printf("Drifting took      %e s \n", tm);
		itime = inc(itime, max_rung);
		iter++;
	}

}
