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

static void memory_transpose() {

   printf( "Beginning memory_transpose test\n");
   particle_set parts(global().opts.nparts);
   timer aos, soa;
   aos.start();
   parts.set_mem_format(particle_set::format::aos);
   aos.stop();
   soa.start();
   parts.set_mem_format(particle_set::format::soa);
   soa.stop();
   printf( "soa to aos time = %e s\n", aos.read());
   printf( "aos to soa time = %e s\n", soa.read());
   printf( "total      time = %e s\n", aos.read() +  soa.read());
}

static void local_sort_random() {
}

void test_run(const std::string test) {
   if (test == "memory_transpose") {
      memory_transpose();
   } else {
      printf("%s is an unknown test.\n", test.c_str());
   }
}
