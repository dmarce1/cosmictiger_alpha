/*
 * global.cpp
 *
 *  Created on: Jan 24, 2021
 *      Author: dmarce1
 */

#include <cosmictiger/global.hpp>

void global_init() {
   global().nparts = 1024;
}

global_t& global() {
   static global_t glob;
   return glob;
}
