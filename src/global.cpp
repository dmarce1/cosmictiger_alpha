/*
 * global.cpp
 *
 *  Created on: Jan 24, 2021
 *      Author: dmarce1
 */

#include <cosmictiger/global.hpp>
#include <cosmictiger/hpx.hpp>


HPX_PLAIN_ACTION (global_init);
HPX_PLAIN_ACTION (global_set_options);

#include <fenv.h>

static global_t glob;

void global_init(options opts, cuda_properties cuda) {
   const auto mychildren = hpx_child_localities();
   hpx::future<void> left, right;
   if (mychildren.first != hpx::invalid_id) {
      left = hpx::async < global_init_action > (mychildren.first, opts, cuda);
   }
   if (mychildren.second != hpx::invalid_id) {
      right = hpx::async < global_init_action > (mychildren.second, opts, cuda);
   }
   glob.opts = opts;
   glob.cuda = cuda;
   if (left.valid()) {
      left.get();
   }
   if (right.valid()) {
      right.get();
   }
 //  feenableexcept (FE_INVALID);
 //  feenableexcept (FE_OVERFLOW);
}

void global_set_options(options opts) {
   const auto mychildren = hpx_child_localities();
   hpx::future<void> left, right;
   if (mychildren.first != hpx::invalid_id) {
      left = hpx::async < global_set_options_action > (mychildren.first, opts);
   }
   if (mychildren.second != hpx::invalid_id) {
      right = hpx::async < global_set_options_action > (mychildren.second, opts);
   }
   glob.opts = opts;
   if (left.valid()) {
      left.get();
   }
   if (right.valid()) {
      right.get();
   }
}

const global_t& global() {
   return glob;
}
