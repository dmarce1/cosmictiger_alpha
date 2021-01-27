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

static void tree_test() {
   particle_set::generate_random_particle_set();
   {
      timer tm;
      printf("Starting worse case tree test\n");
      tm.start();
      tree root;
      tm.stop();
      printf("Tree test took %e s\n", tm.read());
   }
   {
      timer tm;
      printf("Starting best case tree test\n");
      tm.start();
      tree root;
      tm.stop();
      printf("Tree test took %e s\n", tm.read());
   }
}

static void local_sort_random() {
   particle_set::generate_random_particle_set();
   range box;
   for (int dim = 0; dim < NDIM; dim++) {
      box.begin[dim] = fixed32::min();
      box.end[dim] = fixed32::max();
   }

   for (int depth = 9; depth < 25; depth++) {
      particle_set::generate_random_particle_set();
      timer tm1, tm2;
      auto set = particle_set::local_particle_set();
      tm1.start();
      particle_set::radix_sort(0, global().opts.nparts, box, 0, depth);
      tm1.stop();
      for (int i = 0; i < global().opts.nparts / 200; i++) {
         int j = rand() % global().opts.nparts;
         int k = rand() % global().opts.nparts;
         auto tmp = set.get_part(j);
         set.set_part(set.get_part(k), j);
         set.set_part(tmp, k);
      }
      tm2.start();
      particle_set::radix_sort(0, global().opts.nparts, box, 0, depth);
      tm2.stop();
      printf("%i %e %e %e %e \n", depth, tm1.read(), tm1.read() / depth, tm2.read(), tm2.read() / depth);
   }
}

void test_run(const std::string test) {
   if (test == "local_sort_random") {
      local_sort_random();
   } else if (test == "tree_test") {
      tree_test();
   } else {
      printf("%s is an unknown test.\n", test.c_str());
   }
}
