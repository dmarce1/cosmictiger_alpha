/*
 * global.cpp
 *
 *  Created on: Jan 24, 2021
 *      Author: dmarce1
 */

#include <cosmictiger/global.hpp>
#include <cosmictiger/hpx.hpp>



static global_t glob;

void global_init(options opts) {
   glob.opts = opts;
}


const global_t& global() {
   return glob;
}
