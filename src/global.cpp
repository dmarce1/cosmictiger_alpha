/*
 * global.cpp
 *
 *  Created on: Jan 24, 2021
 *      Author: dmarce1
 */

#include <cosmictiger/global.hpp>
#include <cosmictiger/hpx.hpp>

HPX_PLAIN_ACTION (global_init);

void global_init(options opts, cuda_properties cuda) {
   const auto mychildren = hpx_child_localities();
   hpx::future<void> left, right;
   if (mychildren.first != hpx::invalid_id) {
      left = hpx::async < global_init_action > (mychildren.first, opts, cuda);
   }
   if (mychildren.first != hpx::invalid_id) {
      right = hpx::async < global_init_action > (mychildren.second, opts, cuda);
   }
   global().opts = opts;
   global().cuda = cuda;
   if (left.valid()) {
      left.get();
   }
   if (right.valid()) {
      right.get();
   }
}

global_t& global() {
   static global_t glob;
   return glob;
}
