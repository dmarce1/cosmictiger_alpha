/*
 * global.hpp
 *
 *  Created on: Jan 24, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_GLOBAL_HPP_
#define COSMICTIGER_GLOBAL_HPP_

#include <cosmictiger/options.hpp>

#include <cstdint>

struct global_t {
   options opts;
};


void global_init(options opts);
global_t& global();

#endif /* COSMICTIGER_GLOBAL_HPP_ */
