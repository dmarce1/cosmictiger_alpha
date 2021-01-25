/*
 * tests.cpp
 *
 *  Created on: Jan 25, 2021
 *      Author: dmarce1
 */

#include <cosmictiger/tests.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/particle_sort.hpp>

static void local_sort_random() {
   timer tm;
   tm.start();
   particle_set::generate_random_particle_set();
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
