/*
 * tests.cpp
 *
 *  Created on: Jan 25, 2021
 *      Author: dmarce1
 */

#include <cosmictiger/kick_return.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/tests.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/particle.hpp>
#include <cosmictiger/drift.hpp>
#include <cosmictiger/sort.hpp>
#include <cmath>

static void tree_test() {
	printf("Doing tree test\n");
	printf("Generating particles\n");
	particle_set parts(global().opts.nparts);
	parts.generate_random();
	tree::set_particle_set(&parts);

	printf("Sorting\n");
	{
		timer tm;
		tm.start();
		tree root;
		root.sort();
		managed_allocator<sort_params>::cleanup();
		tm.stop();
		managed_allocator<multipole>::cleanup();
		managed_allocator<tree>::cleanup();
		printf("Done sorting in %e\n", tm.read());
	}
	{
		timer tm;
		tm.start();
		tree root;
		root.sort();
		tm.stop();
		printf("Done sorting in %e\n", tm.read());
	}
}

void kick_test() {
	printf("Doing kick test\n");
	printf("Generating particles\n");
	particle_set parts(global().opts.nparts);
	parts.load_particles("ics");
	// parts.generate_grid();
	tree::set_particle_set(&parts);
	particle_set *parts_ptr;
	CUDA_MALLOC(parts_ptr, sizeof(particle_set));
	new (parts_ptr) particle_set(parts.get_virtual_particle_set());
	tree::cuda_set_kick_params(parts_ptr);
	for (int i = 0; i < 2; i++) {
		tree root;
		timer tm_sort, tm_kick, tm_cleanup;
		tm_sort.start();
		root.sort();
		tm_sort.stop();
		tm_kick.start();
		tree_ptr root_ptr;
		root_ptr.ptr = (uintptr_t) &root;
		//  root_ptr.rank = hpx_rank();
		// printf( "%li", size_t(WORKSPACE_SIZE));
		kick_params_type *params_ptr;
		CUDA_MALLOC(params_ptr, 1);
		new (params_ptr) kick_params_type();
		params_ptr->dchecks.push(root_ptr);
		params_ptr->echecks.push(root_ptr);

		// printf( "---------> %li %li\n", root_ptr.ptr, dchecks[0].ptr);
		array<fixed32, NDIM> Lpos;
		expansion<float> L;
		for (int i = 0; i < LP; i++) {
			L[i] = 0.f;
		}
		for (int dim = 0; dim < NDIM; dim++) {
			Lpos[dim] = 0.5;
		}
		kick_return_init(0);
		params_ptr->L[0] = L;
		params_ptr->Lpos[0] = Lpos;
		params_ptr->t0 = true;
		root.kick(params_ptr).get();
		tm_kick.stop();
		/*   tm.start();
		 drift(parts_ptr, 1.0,1.0,1.0);
		 tm.stop();
		 printf( "Drift took %e s\n", tm.read());*/
		tree::cleanup();
		tm_cleanup.start();
		managed_allocator<sort_params>::cleanup();
		managed_allocator<multipole>::cleanup();
		managed_allocator<tree>::cleanup();
		params_ptr->kick_params_type::~kick_params_type();
		CUDA_FREE(params_ptr);
		tm_cleanup.stop();
		const auto total = tm_sort.read() + tm_kick.read() + tm_cleanup.read();
		kick_return_show();
		/*   printf("PP/part = %f\n", get_pp_inters());
		 printf("PC/part = %f\n", get_pc_inters());
		 printf("CP/part = %f\n", get_cp_inters());
		 printf("CC/part = %f\n", get_cc_inters());*/
		printf("Sort    = %e s\n", tm_sort.read());
		printf("Kick    = %e s\n", tm_kick.read());
		printf("Cleanup = %e s\n", tm_cleanup.read());
		printf("Total   = %e s\n", total);
		//  printf("GFLOP   = %e s\n", rc.flops / 1024. / 1024. / 1024.);
		// printf("GFLOP/s = %e\n", rc.flops / 1024. / 1024. / 1024. / total);
	}
	parts_ptr->particle_set::~particle_set();
	CUDA_FREE(parts_ptr);
}

void drift_test() {
	printf("Doing kick test\n");
	printf("Generating particles\n");
	particle_set parts(global().opts.nparts);
	parts.load_particles("ics");
	// parts.generate_grid();
	tree::set_particle_set(&parts);
	particle_set *parts_ptr;
	CUDA_MALLOC(parts_ptr, sizeof(particle_set));
	new (parts_ptr) particle_set(parts.get_virtual_particle_set());
	tree::cuda_set_kick_params(parts_ptr);
	for (int i = 0; i < 2; i++) {
		tree root;
		timer tm_sort, tm_kick, tm_cleanup;
		tm_sort.start();
		root.sort();
		tm_sort.stop();
		tm_kick.start();
		tree_ptr root_ptr;
		root_ptr.ptr = (uintptr_t) &root;
		//  root_ptr.rank = hpx_rank();
		// printf( "%li", size_t(WORKSPACE_SIZE));
		kick_params_type *params_ptr;
		CUDA_MALLOC(params_ptr, 1);
		new (params_ptr) kick_params_type();
		params_ptr->dchecks.push(root_ptr);
		params_ptr->echecks.push(root_ptr);

		// printf( "---------> %li %li\n", root_ptr.ptr, dchecks[0].ptr);
		array<fixed32, NDIM> Lpos;
		expansion<float> L;
		for (int i = 0; i < LP; i++) {
			L[i] = 0.f;
		}
		for (int dim = 0; dim < NDIM; dim++) {
			Lpos[dim] = 0.5;
		}
		kick_return_init(0);
		params_ptr->L[0] = L;
		params_ptr->Lpos[0] = Lpos;
		params_ptr->t0 = true;
		root.kick(params_ptr).get();
		tm_kick.stop();
		/*   tm.start();
		 drift(parts_ptr, 1.0,1.0,1.0);
		 tm.stop();
		 printf( "Drift took %e s\n", tm.read());*/
		tree::cleanup();
		tm_cleanup.start();
		managed_allocator<sort_params>::cleanup();
		managed_allocator<multipole>::cleanup();
		managed_allocator<tree>::cleanup();
		tm_cleanup.stop();
		const auto total = tm_sort.read() + tm_kick.read() + tm_cleanup.read();
		kick_return_show();
		timer tm_drift;
		printf("Doing drift\n");
		tm_drift.start();
		const int max_rung = kick_return_max_rung();
		double dt = params_ptr->t0 / (1 << max_rung);
		printf("Maximum rung = %i timestep = %e\n", max_rung, dt);
		drift_particles(parts.get_virtual_particle_set(), dt, 1.0, 1.0);
		tm_drift.stop();
		printf("Drift took %e seconds\n", tm_drift.read());

		params_ptr->kick_params_type::~kick_params_type();
		CUDA_FREE(params_ptr);

		/*   printf("PP/part = %f\n", get_pp_inters());
		 printf("PC/part = %f\n", get_pc_inters());
		 printf("CP/part = %f\n", get_cp_inters());
		 printf("CC/part = %f\n", get_cc_inters());*/
		printf("Sort    = %e s\n", tm_sort.read());
		printf("Kick    = %e s\n", tm_kick.read());
		printf("Cleanup = %e s\n", tm_cleanup.read());
		printf("Total   = %e s\n", total);
		//  printf("GFLOP   = %e s\n", rc.flops / 1024. / 1024. / 1024.);
		// printf("GFLOP/s = %e\n", rc.flops / 1024. / 1024. / 1024. / total);
	}
	parts_ptr->particle_set::~particle_set();
	CUDA_FREE(parts_ptr);
}

#ifdef TEST_FORCE
void force_test() {
	printf("Doing force test\n");
	printf("Generating particles\n");
	particle_set parts(global().opts.nparts);
	// parts.load_particles("ics");
	parts.generate_random();
	tree::set_particle_set(&parts);
	particle_set *parts_ptr;
	CUDA_MALLOC(parts_ptr, sizeof(particle_set));
	new (parts_ptr) particle_set(parts.get_virtual_particle_set());
	tree::cuda_set_kick_params(parts_ptr);
	tree root;
	root.sort();
	tree_ptr root_ptr;
	root_ptr.ptr = (uintptr_t) & root;
	//  root_ptr.rank = hpx_rank();
	// printf( "%li", size_t(WORKSPACE_SIZE));
	kick_params_type *params_ptr;
	CUDA_MALLOC(params_ptr, 1);
	new (params_ptr) kick_params_type();
	params_ptr->dchecks.push(root_ptr);
	params_ptr->echecks.push(root_ptr);

	// printf( "---------> %li %li\n", root_ptr.ptr, dchecks[0].ptr);
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
	params_ptr->t0 = true;
	kick_return_init(0);
	root.kick(params_ptr).get();
	kick_return_show();
	tree::cleanup();
	managed_allocator<tree>::cleanup();
	params_ptr->kick_params_type::~kick_params_type();
	timer tm;
	tm.start();
	printf( "Doing comparison\n");
	cuda_compare_with_direct(parts_ptr);
	tm.stop();
	printf( "Comparison took %e s\n", tm.read());
	// printf("GFLOP   = %e s\n", rc.flops / 1024. / 1024. / 1024.);
	CUDA_FREE(params_ptr);
	CUDA_FREE(parts_ptr);
}
#endif

static void sort() {

	timer tm;
	particle_set parts(global().opts.nparts);
	parts.generate_random();

	timer tm1;
	tm1.start();
	find_median(parts.get_virtual_particle_set(),0, parts.size(), fixed32(0.0), fixed32::max(), 0);
	tm1.stop();
	printf("%e\n", tm1.read());
	tm1.reset();
	tm1.start();
	find_median(parts.get_virtual_particle_set(),0, parts.size(), fixed32(0.0), fixed32::max(), 0);
	tm1.stop();

	printf("%e\n", tm1.read());

}

void test_run(const std::string test) {
	if (test == "sort") {
		sort();
	} else if (test == "tree_test") {
		tree_test();
	} else if (test == "drift") {
		drift_test();
	} else if (test == "kick") {
		kick_test();
#ifdef TEST_FORCE
	} else if (test == "force") {
		force_test();
#endif
	} else {
		printf("%s is an unknown test.\n", test.c_str());
	}
}
