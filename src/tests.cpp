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
      tm.stop();
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

      printf("%li %e %e %e %e\n", depth, tm1.read(), tm2.read(),  tm1.read()/depth, tm2.read()/depth);

   }

   printf("Time to sort best case: %e s\n\n", tm.read());
   tm.reset();

}

void test_run(const std::string test) {
   if (test == "sort") {
      sort();
   } else if (test == "tree_test") {
      tree_test();
   } else {
      printf("%s is an unknown test.\n", test.c_str());
   }
}
