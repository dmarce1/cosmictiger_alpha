/*
 * tests.cpp
 *
 *  Created on: Jan 25, 2021
 *      Author: dmarce1
 */

#include <cosmictiger/global.hpp>
#include <cosmictiger/tests.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/particle_sort.hpp>

static void local_sort_random() {
   timer tm;
   particle_set::generate_random_particle_set();

   tm.start();
   particle_sort::sort(0, global().opts.nparts, 1, fixed32(0.00));
   tm.stop();
   printf("Test took %e seconds.\n", tm.read());
}

void test_run(const std::string test) {
   if (test == "local_sort_random") {
      local_sort_random();
   } else {
      printf("%s is an unknown test.\n", test.c_str());
   }
}
