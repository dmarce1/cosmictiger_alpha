/*
 * rand.cpp
 *
 *  Created on: Jan 25, 2021
 *      Author: dmarce1
 */


#include <cosmictiger/rand.hpp>

#include <cosmictiger/kernel.hpp>

using hpxfft::cmplx;


__global__
void generate_random_vectors(fixed32* x, fixed32* y, fixed32* z, size_t N, int seed) {
	const uint64_t mod = 1LL << 31LL;
	const uint64_t a1 = 1664525LL;
	const uint64_t a2 = 22695477LL;
	const uint64_t c1 = 1013904223LL;
	const uint64_t c2 = 1LL;
	const int& thread = threadIdx.x;
	const int& block_size = blockDim.x;
	const auto count = threadIdx.x + blockIdx.x * blockDim.x;
	uint64_t int1 = seed;
	for (int i = 0; i < count; i++) {
		int1 = (a1 * int1 + c1) % mod;
	}
	const size_t start = blockIdx.x * N / gridDim.x;
	const size_t stop = (blockIdx.x + 1) * N / gridDim.x;
	for (size_t i = start + thread; i < stop; i += block_size) {
		int1 = (a2 * (uint64_t) int1 + c2) % mod;
		*((unsigned int*) x + i) = (int1 << 1) & 0xFFFFFFFFLL;
		int1 = (a2 * (uint64_t) int1 + c2) % mod;
		*((unsigned int*) y + i) = (int1 << 1) & 0xFFFFFFFFLL;
		int1 = (a2 * (uint64_t) int1 + c2) % mod;
		*((unsigned int*) z + i) = (int1 << 1) & 0xFFFFFFFFLL;
	}
	__syncthreads();
}




