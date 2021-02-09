/*
 * tests.cpp
 *
 *  Created on: Jan 25, 2021
 *      Author: dmarce1
 */

#include <cosmictiger/global.hpp>
#include <cosmictiger/tests.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/particle.hpp>
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
   parts.generate_random();
   tree::set_particle_set(&parts);
   particle_set* parts_ptr;
   CUDA_MALLOC(parts_ptr,sizeof(particle_set));
   new (parts_ptr) particle_set(parts.get_virtual_particle_set());
   tree::cuda_set_kick_params(parts_ptr, 0.7, 0);
   for (int i = 0; i < 2; i++) {
      tree root;
      timer tm;
      tm.start();
      root.sort();
      tree_ptr root_ptr;
      root_ptr.ptr = (uintptr_t) & root;
      root_ptr.rank = hpx_rank();
      // printf( "%li", size_t(WORKSPACE_SIZE));
      kick_params_type* params_ptr;
      CUDA_MALLOC(params_ptr,1);
      new (params_ptr) kick_params_type();
      params_ptr->dstack.copy_to(&root_ptr,1);
      params_ptr->estack.copy_to(&root_ptr,1);

      // printf( "---------> %li %li\n", root_ptr.ptr, dchecks[0].ptr);
      expansion<accum_real> L;
      for (int i = 0; i < LP; i++) {
         L[i] = 0.f;
      }
      array<accum_real, NDIM> Lpos;
      for (int dim = 0; dim < NDIM; dim++) {
         Lpos[dim] = 0.5;
      }
      params_ptr->L[0] = L;
    //  parts_ptr->set_read_mostly(true);
      root.kick(params_ptr).get();
     // parts_ptr->set_read_mostly(false);
      tm.stop();
      tree::cleanup();
      params_ptr->kick_params_type::~kick_params_type();
      CUDA_FREE(params_ptr);
      printf("Done kicking in %e seconds\n\n", tm.read());
   }
   parts_ptr->particle_set::~particle_set();
   CUDA_FREE(parts_ptr);
}

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
   } else {
      printf("%s is an unknown test.\n", test.c_str());
   }
}
