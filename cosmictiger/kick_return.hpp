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

#define KR_PP 0
#define KR_PC 1
#define KR_CP 2
#define KR_CC 3
#define KR_OP 4
#define KR_EWCC 5

#define KR_COUNT 6


struct kick_return {
	array<double,MAX_RUNG> phis;
	array<int,MAX_RUNG> rung_cnt;
	array<double,KR_COUNT> flop;
	array<double,KR_COUNT> count;
	int min_rung;
};


kick_return kick_return_get();
void kick_return_update_pot_cpu(int rung, float);
__device__ void kick_return_update_pot_gpu(int rung, float);
void kick_return_update_rung_cpu(int rung);
__device__ void kick_return_update_rung_gpu(int rung);
__device__ void kick_return_update_interactions_gpu(int itype, int count, int flops);
void kick_return_update_interactions_cpu(int itype, int count, int flops);


void kick_return_init(int min_rung);


void kick_return_show();

#endif /* KICK_RETURN_HPP_ */
