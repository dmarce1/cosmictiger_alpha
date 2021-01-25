/*
 * hpx.hpp
 *
 *  Created on: Jan 23, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_HPX_HPP_
#define COSMICTIGER_HPX_HPP_

#include <cosmictiger/hpx.hpp>
#include <hpx/hpx.hpp>

void hpx_init();
int hpx_rank();
int hpx_size();
const std::vector<hpx::id_type>& hpx_localities();
const std::pair<hpx::id_type,hpx::id_type>& hpx_child_localities();



#endif /* COSMICTIGER_HPX_HPP_ */


