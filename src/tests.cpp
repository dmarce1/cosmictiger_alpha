/*
 * tests.cpp
 *
 *  Created on: Jan 25, 2021
 *      Author: dmarce1
 */

#include <cosmictiger/global.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/tests.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/particle.hpp>
#include <cmath>

void show_timings();

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
   ewald_indices *real_indices;
   ewald_indices *four_indices;
   periodic_parts *pparts;
   CUDA_MALLOC(real_indices, 1);
   CUDA_MALLOC(four_indices, 1);
   CUDA_MALLOC(pparts, 1);
   new (real_indices) ewald_indices(EWALD_NREAL, false);
   new (four_indices) ewald_indices(EWALD_NFOUR, true);
   new (pparts) periodic_parts();
   tree::cuda_set_kick_params(parts_ptr, real_indices, four_indices, pparts);
   for (int i = 0; i < 2; i++) {
      tree root;
      timer tm_sort, tm_kick, tm_cleanup;
      tm_sort.start();
      root.sort();
      tm_sort.stop();
      tm_kick.start();
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
      auto rc = root.kick(params_ptr).get();
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
      printf("PP/part = %f\n", get_pp_inters());
      printf("PC/part = %f\n", get_pc_inters());
      printf("CP/part = %f\n", get_cp_inters());
      printf("CC/part = %f\n", get_cc_inters());
      printf("Sort    = %e s\n", tm_sort.read());
      printf("Kick    = %e s\n", tm_kick.read());
      printf("Cleanup = %e s\n", tm_cleanup.read());
      printf("Total   = %e s\n", total);
      printf("GFLOP   = %e s\n", rc.flops / 1024. / 1024. / 1024.);
      printf("GFLOP/s = %e\n", rc.flops / 1024. / 1024. / 1024. / total);
      show_timings();
   }
   parts_ptr->particle_set::~particle_set();
   CUDA_FREE(parts_ptr);
}

#ifdef TEST_FORCE
void force_test() {
   printf("Doing force test\n");
   printf("Generating particles\n");
   particle_set parts(global().opts.nparts);
   parts.generate_random();
   tree::set_particle_set(&parts);
   particle_set *parts_ptr;
   CUDA_MALLOC(parts_ptr, sizeof(particle_set));
   new (parts_ptr) particle_set(parts.get_virtual_particle_set());
   ewald_indices *real_indices;
   ewald_indices *four_indices;
   periodic_parts *pparts;
   CUDA_MALLOC(real_indices, 1);
   CUDA_MALLOC(four_indices, 1);
   CUDA_MALLOC(pparts, 1);
   new (real_indices) ewald_indices(EWALD_NREAL, false);
   new (four_indices) ewald_indices(EWALD_NFOUR, true);
   new (pparts) periodic_parts();
   tree::cuda_set_kick_params(parts_ptr, real_indices, four_indices, pparts);
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
   auto rc = root.kick(params_ptr).get();
   tree::cleanup();
   managed_allocator<tree>::cleanup();
   params_ptr->kick_params_type::~kick_params_type();
   timer tm;
   tm.start();
   printf( "Doing comparison\n");
   cuda_compare_with_direct(parts_ptr);
   tm.stop();
   printf( "Comparison took %e s\n", tm.read());
   printf("GFLOP   = %e s\n", rc.flops / 1024. / 1024. / 1024.);
   CUDA_FREE(params_ptr);
   CUDA_FREE(parts_ptr);
}
#endif

static void sort() {
   timer tm;
   particle_set parts(global().opts.nparts);
   parts.generate_random();

   for (int depth = 3; depth < 30; depth += 3) {
      timer tm1;
      tm1.start();
      parts.local_sort(0, global().opts.nparts, depth, 0, 1 << depth);
      tm1.stop();

      timer tm2;
      tm2.start();
      parts.local_sort(0, global().opts.nparts, depth, 0, 1 << depth);
      tm2.stop();

      printf("%i %e %e %e %e\n", depth, tm1.read(), tm2.read(), tm1.read() / depth, tm2.read() / depth);

   }

   printf("Time to sort best case: %e s\n\n", tm.read());
   tm.reset();

}

void test_run(const std::string test) {
   if (test == "sort") {
      sort();
   } else if (test == "tree_test") {
      tree_test();
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
