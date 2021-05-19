/*
 * hpx.hpp
 *
 *  Created on: Jan 23, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_HPX_HPP_
#define COSMICTIGER_HPX_HPP_

#ifndef __CUDACC__
#ifdef USE_HPX
#include <hpx/hpx.hpp>
#include <hpx/async.hpp>
#include <hpx/hpx_finalize.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/serialization.hpp>
#else
#include <hpx_lite/hpx/hpx_lite.hpp>
#endif

using mutex_type = hpx::lcos::local::spinlock;
using spinlock_type = hpx::lcos::local::spinlock;

const std::vector<hpx::id_type>& hpx_localities();
const std::pair<hpx::id_type,hpx::id_type>& hpx_child_localities();
#endif
void hpx_init();
int hpx_rank();
int hpx_size();
int hardware_concurrency();


#endif /* COSMICTIGER_HPX_HPP_ */


