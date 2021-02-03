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
   printf("Sorting\n");
   tree root;
   timer tm;
   tm.start();
   root.sort();
   tm.stop();
   printf("Sort in %e seconds\n", tm.read());
   tm.reset();
   tree_ptr root_ptr;
   root_ptr.ptr = (uintptr_t) & root;
   root_ptr.rank = hpx_rank();
   // printf( "%li", size_t(WORKSPACE_SIZE));
   kick_stack stack;
   stack.dchecks.resize(TREE_MAX_DEPTH);
   stack.echecks.resize(TREE_MAX_DEPTH);
   stack.dchecks[0].push_back(root_ptr);
   stack.echecks[0].push_back(root_ptr);
   // printf( "---------> %li %li\n", root_ptr.ptr, dchecks[0].ptr);
   expansion L;
   for (int i = 0; i < LP; i++) {
      L[i] = 0.f;
   }
   array<exp_real, NDIM> Lpos;
   for (int dim = 0; dim < NDIM; dim++) {
      Lpos[dim] = 0.5;
   }
   tree::set_kick_parameters(0.7, 0);
   printf("Kicking\n");
   tm.start();
   root.kick(L, Lpos, stack, 0);
   tm.stop();
   printf("Done kicking in %e seconds\n", tm.read());
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
