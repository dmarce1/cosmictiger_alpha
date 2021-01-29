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



static void sort() {
   timer tm;
   particle_set parts(global().opts.nparts);
   parts.generate_random();

   const size_t depth = (int(std::log(global().opts.nparts / global().opts.bucket_size) / std::log(8)) + 1)*NDIM;

   printf( "Using %li levels\n", depth);

   tm.start();
   parts.local_sort(0, global().opts.nparts, 18, 0, 1<<18);
   tm.stop();
   printf("Time to sort worst case: %e s\n", tm.read());
   tm.reset();

   tm.start();
   parts.local_sort(0, global().opts.nparts,  18, 0, 1<<18);
   tm.stop();
   printf("Time to sort best case: %e s\n", tm.read());
   tm.reset();

   tm.start();
   parts.local_sort(0, global().opts.nparts, 18, 0, 1<<18);
   tm.reset();
   tm.stop();
   printf("Time to sort best case: %e s\n", tm.read());

}

void test_run(const std::string test) {
   if (test == "sort") {
      sort();
   } else {
      printf("%s is an unknown test.\n", test.c_str());
   }
}
