/*
 * kick_return.hpp
 *
 *  Created on: Mar 8, 2021
 *      Author: dmarce1
 */

#ifndef KICK_RETURN_HPP_
#define KICK_RETURN_HPP_

#include <cosmictiger/defs.hpp>
#include <cosmictiger/array.hpp>

struct kick_return {
	array<int,MAX_RUNG> rung_cnt;
	int min_rung;
};


kick_return kick_return_get();
void kick_return_update_rung_cpu(int rung);
__device__ void kick_return_update_rung_gpu(int rung);
void kick_return_init(int min_rung);


void kick_return_show();

#endif /* KICK_RETURN_HPP_ */
