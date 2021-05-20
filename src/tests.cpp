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
#include <cosmictiger/particle_server.hpp>
#include <cosmictiger/drift.hpp>
#include <cosmictiger/groups.hpp>
#include <cosmictiger/time.hpp>
#include <cmath>

static void psort_test() {
	particle_server pserv;
	pserv.init();
	timer tm;
	pserv.generate_random();
	tree_data_initialize(TREE_KICK);
	printf("Starting\n");
	tm.start();
	pserv.apply_domain_decomp();
	tm.stop();
	printf("took %e s\n", tm.read());
	tm.reset();
	tm.start();

//	pserv.check_domain_bounds();

	sort_params params;
	params.min_rung = 0;
	params.theta = global().opts.theta;
	params.group_sort = false;
	tree::sort(params);
	tm.stop();
	printf("Done sorting in %e\n", tm.read());

	printf("took %e s\n", tm.read());
}

static void tree_test() {
	printf("Doing tree test\n");
	printf("Generating particles\n");
	particle_set parts(global().opts.nparts);
	parts.generate_random(1243);

	printf("Sorting\n");
	{
		timer tm;
		tm.start();
		tree root;
		sort_params params;
		params.min_rung = 0;
		params.group_sort = false;
		params.theta = global().opts.theta;
		tree_data_initialize_kick();
		root.sort(params);
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
		sort_params params;
		params.group_sort = false;
		params.min_rung = 0;
		params.theta = global().opts.theta;
		root.sort(params);
		tm.stop();
		printf("Done sorting in %e\n", tm.read());
	}
}
void kick_test() {
	printf("Doing kick test\n");
	printf("Generating particles\n");
	particle_server pserv;
	pserv.init();
	pserv.generate_random();
	timer ttime;
	std::vector<double> timings;

	for (int i = 0; i < NKICKS + 1; i++) {
		tree_data_initialize(TREE_KICK);
		printf("Kick %i\n", i);
		ttime.start();
		tree root;
		timer tm_sort, tm_kick[2], tm_cleanup;
		tm_sort.start();
		sort_params params;
		params.group_sort = false;
		params.theta = global().opts.theta;
		params.min_rung = 0;
		tree_ptr root_ptr;
		pserv.apply_domain_decomp();
		root_ptr = root.sort(params).check;
		tm_sort.stop();
		//  root_ptr.rank = hpx_rank();
		// printf( "%li", size_t(WORKSPACE_SIZE));

		kick_return_init(0);
		for (int pass = 0; pass < 2; pass++) {
			printf( "pass %i\n", pass);
			tm_kick[pass].start();
			if (hpx_size() == 1 && pass == 0) {
				continue;
			}
			kick_params_type *params_ptr;
			CUDA_MALLOC(params_ptr, 1);
			new (params_ptr) kick_params_type();
			params_ptr->dry_run = (pass == 0);
			params_ptr->tptr = root_ptr;
			params_ptr->dchecks.push(root_ptr);
			params_ptr->echecks.push(root_ptr);
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
			params_ptr->t0 = 1;
			params_ptr->full_eval = false;
			params_ptr->rung = 0;
			root.kick(params_ptr).get();
			params_ptr->kick_params_type::~kick_params_type();
			CUDA_FREE(params_ptr);
			tm_kick[pass].stop();
//			printf( "Returning\n");
//			return;
		}
		tm_cleanup.start();
		tree::cleanup();
		tm_cleanup.stop();
		const auto total = tm_sort.read() + tm_kick[0].read() + tm_kick[1].read() + tm_cleanup.read();
//		kick_return_show();
		printf("Sort         = %e s\n", tm_sort.read());
		printf("Bounds       = %e s\n", tm_kick[0].read());
		printf("Kick         = %e s\n", tm_kick[1].read());
		printf("Cleanup      = %e s\n", tm_cleanup.read());
		printf("Total Score  = %e s\n", total);
		//  printf("GFLOP   = %e s\n", rc.flops / 1024. / 1024. / 1024.);
		// printf("GFLOP/s = %e\n", rc.flops / 1024. / 1024. / 1024. / total);
		//	tree::show_timings();
		ttime.stop();
		if (i > 0) {
			timings.push_back(tm_kick[0].read() + tm_kick[1].read() + tm_sort.read());
		}
		ttime.reset();
	}
	double avg = 0.0, dev = 0.0;
	for (int i = 0; i < NKICKS; i++) {
		avg += timings[i];
	}
	avg /= NKICKS;
	for (int i = 0; i < NKICKS; i++) {
		dev += sqr(avg - timings[i]);
	}
	dev = std::sqrt(dev / NKICKS);
	printf("---- Bucket Size = %i, Score = %e +/- %e\n", global().opts.bucket_size, avg, dev);
	FILE* fp = fopen("timings.dat", "at");
	fprintf(fp, "%i %e %e\n", global().opts.bucket_size, avg, dev);
	fclose(fp);
}

void group_test() {
	printf("Doing group test\n");
	printf("Generating particles\n");
	particle_set parts(global().opts.nparts);
	parts.generate_random(1234);
	particle_set *parts_ptr;
	CUDA_MALLOC(parts_ptr, sizeof(particle_set));
	new (parts_ptr) particle_set(parts.get_virtual_particle_set());
	timer ttime;
	std::vector<double> timings;
	printf("Finding Groups\n");
	ttime.start();
	tree root;
	timer tm_sort, tm_kick, tm_cleanup;
	tm_sort.start();
	sort_params params;
	params.theta = global().opts.theta;
	params.min_rung = 0;
	tree_ptr root_ptr;
	parts.init_groups();
	params.group_sort = true;
	parts.init_groups();
	printf("Sorting\n");
	tree_data_initialize_groups();
	root_ptr = root.sort(params).check;
	printf("Done Sorting\n");
	tm_sort.stop();
	tm_kick.start();
	//  root_ptr.rank = hpx_rank();
	// printf( "%li", size_t(WORKSPACE_SIZE));
	group_param_type *params_ptr;
	CUDA_MALLOC(params_ptr, 1);
	new (params_ptr) group_param_type();
	params_ptr->self = root_ptr;
	params_ptr->link_len = 1.0 / pow(global().opts.nparts, 1.0 / 3.0) / 2.0;
	params_ptr->parts = parts.get_virtual_particle_set();
	params_ptr->checks.push(root_ptr);
	params_ptr->first_round = true;
	printf("Searching\n");
	fflush(stdout);
	timer tm;
	tm.start();
	tree_database_set_groups();
//	find_groups_phase1(params_ptr).get();
	while (find_groups(params_ptr).get()) {
		tree_database_set_groups();
		printf("Iterating\n");
		params_ptr->self = root_ptr;
		params_ptr->checks.resize(0);
		params_ptr->checks.push(root_ptr);
		params_ptr->first_round = false;
	}
	tm_kick.stop();
	tree::cleanup();
	CUDA_FREE(params_ptr);
	tm.stop();
	printf("Groups found in %e s\n", tm.read());
	tm.reset();
	tm.start();
	/*	group_data_create(parts);
	 group_data_reduce();
	 group_data_output(stdout);
	 group_data_destroy();*/
	tm.stop();
	printf("Table created in %e s\n", tm.read());

}

void drift_test() {
	/*	printf("Doing kick test\n");
	 printf("Generating particles\n");
	 particle_set parts(global().opts.nparts);
	 parts.load_particles("ics");
	 // parts.generate_grid();
	 particle_set *parts_ptr;
	 CUDA_MALLOC(parts_ptr, sizeof(particle_set));
	 new (parts_ptr) particle_set(parts.get_virtual_particle_set());
	 for (int i = 0; i < 2; i++) {
	 tree root;
	 timer tm_sort, tm_kick, tm_cleanup;
	 tm_sort.start();
	 sort_params params;
	 params.theta = global().opts.theta;
	 params.min_rung = 0;
	 tree_data_initialize_kick();
	 root.sort(params);
	 tm_sort.stop();
	 tm_kick.start();
	 tree_ptr root_ptr;

	 root_ptr.dindex = 0;
	 //  root_ptr.rank = hpx_rank();
	 // printf( "%li", size_t(WORKSPACE_SIZE));
	 kick_params_type *params_ptr;
	 CUDA_MALLOC(params_ptr, 1);
	 new (params_ptr) kick_params_type();
	 params_ptr->dchecks.push(root_ptr);
	 params_ptr->echecks.push(root_ptr);
	 params_ptr->full_eval = true;

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
	 double a, b, c, d;
	 drift_particles(parts.get_virtual_particle_set(), dt, 1.0, &a, &b, &c, &d, 0.0, 1.0);
	 tm_drift.stop();
	 printf("Drift took %e seconds\n", tm_drift.read());

	 params_ptr->kick_params_type::~kick_params_type();
	 CUDA_FREE(params_ptr);
	 printf("Sort    = %e s\n", tm_sort.read());
	 printf("Kick    = %e s\n", tm_kick.read());
	 printf("Cleanup = %e s\n", tm_cleanup.read());
	 printf("Total   = %e s\n", total);
	 //  printf("GFLOP   = %e s\n", rc.flops / 1024. / 1024. / 1024.);
	 // printf("GFLOP/s = %e\n", rc.flops / 1024. / 1024. / 1024. / total);
	 }
	 parts_ptr->particle_set::~particle_set();
	 CUDA_FREE(parts_ptr);*/
}

#ifdef TEST_FORCE
void force_test() {
	printf("Doing force test\n");
	printf("Generating particles\n");
	particle_server pserv;
	pserv.init();
	pserv.generate_random();
	tree root;
	sort_params params;
	params.theta = global().opts.theta;
	params.min_rung = 0;
	tree_ptr root_ptr;

	tree_data_initialize_kick();
	root_ptr = root.sort(params).check;
	//  root_ptr.rank = hpx_rank();
	// printf( "%li", size_t(WORKSPACE_SIZE));
	kick_params_type *params_ptr;
	CUDA_MALLOC(params_ptr, 1);
	new (params_ptr) kick_params_type();
	params_ptr->tptr = root_ptr;
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
	params_ptr->full_eval = true;
	kick_return_init(0);
	params_ptr->theta = 0.4;
	root.kick(params_ptr).get();
	kick_return_show();
	tree::cleanup();
	managed_allocator<tree>::cleanup();
	params_ptr->kick_params_type::~kick_params_type();
	timer tm;
	tm.start();
	printf("Doing comparison\n");
	cuda_compare_with_direct(&pserv.get_particle_set());
	tm.stop();
	printf("Comparison took %e s\n", tm.read());
	// printf("GFLOP   = %e s\n", rc.flops / 1024. / 1024. / 1024.);
	CUDA_FREE(params_ptr);
}
#endif

static void sort() {

	timer tm;
	particle_set parts(global().opts.nparts);

	timer tm1;
	for (int i = global().opts.nparts; i >= 64; i /= 2) {
		parts.generate_random(1234);
		tm1.start();
		//	parts.sort_range(0, i, 0.5, 0);
		tm1.stop();
		printf("%i %e\n", i, tm1.read());
		tm1.reset();
	}

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
	} else if (test == "psort") {
		psort_test();
	} else if (test == "group") {
		group_test();
#ifdef TEST_FORCE
	} else if (test == "force") {
		force_test();
#endif
	} else {
		printf("%s is an unknown test.\n", test.c_str());
	}
}
