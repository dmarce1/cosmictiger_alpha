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
#include <cosmictiger/particle_sort.hpp>

static void local_sort_random() {
   timer tm;
   particle_set::generate_random_particle_set();

   tm.start();
   const auto xmid = particle_set::sort(0, global().opts.nparts, 1, 0.0);
   printf("Test took %e seconds.\n", tm.read());
   const auto parts = particle_set::local_particle_set();

   printf("Checking results\n");
   bool success = true;
   for (int i = 0; i < global().opts.nparts; i++) {
      const auto x = parts.get_x(i, 1);
      if (x > fixed32(0) && i < xmid) {
         printf("hi in lo %e %e\n", x.to_float(), 0.0f);
         success = false;
         break;
      } else if (x < fixed32(0) && i >= xmid) {
         printf("lo in hi %e %e\n", x.to_float(), 0.0f);
         success = false;
         break;
      }
   }
   if (success) {
      printf("Passed\n");
   } else {
      printf("Failed\n");
   }

   tm.stop();
}

void test_run(const std::string test) {
   if (test == "local_sort_random") {
      local_sort_random();
   } else {
      printf("%s is an unknown test.\n", test.c_str());
   }
}
