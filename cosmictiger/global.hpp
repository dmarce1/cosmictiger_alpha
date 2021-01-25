/*
 * global.hpp
 *
 *  Created on: Jan 24, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_GLOBAL_HPP_
#define COSMICTIGER_GLOBAL_HPP_

#include <cstdint>

struct global_t {
   int64_t nparts;
};


void global_init();
global_t& global();

#endif /* COSMICTIGER_GLOBAL_HPP_ */
