/*
 * hpx.hpp
 *
 *  Created on: Jan 23, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_HPX_HPP_
#define COSMICTIGER_HPX_HPP_

#ifndef __CUDACC__
#include <cosmictiger/hpx.hpp>
#include <hpx/hpx.hpp>
#include <hpx/async.hpp>
#include <hpx/hpx_finalize.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/components.hpp>
#include <hpx/collectives/broadcast_direct.hpp>

using mutex_type = hpx::lcos::local::mutex;

void hpx_init();
int hpx_rank();
int hpx_size();
const std::vector<hpx::id_type>& hpx_localities();
const std::vector<hpx::id_type>& hpx_remote_localities();
const std::pair<hpx::id_type,hpx::id_type>& hpx_child_localities();
#endif


#endif /* COSMICTIGER_HPX_HPP_ */


