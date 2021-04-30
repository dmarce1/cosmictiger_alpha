#include <cosmictiger/math.hpp>

#include <curand_kernel.h>

double find_root(std::function<double(double)> f) {
	double x = 0.5;
	double err;
	int iters = 0;
	do {
		double dx0 = x * 1.0e-6;
		if (abs(dx0) == 0.0) {
			dx0 = 1.0e-10;
		}
		double fx = f(x);
		double dfdx = (f(x + dx0) - fx) / dx0;
		double dx = -fx / dfdx;
		err = abs(dx / max(1.0, abs(x)));
		x += 0.5 * dx;
		iters++;
		if (iters > 1000000) {
			printf("Finished early with error = %e\n", err);
			break;
		}
	} while (err > 1.0e-6);
	return x;
}

__global__
void generate_random_normals(cmplx* nums, size_t N, int seed) {
	const uint64_t mod = 1LL << 31LL;
	const uint64_t a1 = 1664525LL;
	const uint64_t a2 = 22695477LL;
	const uint64_t a3 = 134775813LL;
	const uint64_t a4 = 214013LL;
	const uint64_t a5 = 16843009LL;
	const uint64_t c1 = 1013904223LL;
	const uint64_t c2 = 1LL;
	const uint64_t c3 = 1LL;
	const uint64_t c4 = 2531011LL;
	const uint64_t c5 = 826366247LL;
	const int& thread = threadIdx.x;
	const int& block_size = blockDim.x;
	const auto count = threadIdx.x + blockIdx.x * blockDim.x;
	uint64_t int1 = seed;
	uint64_t int2 = (a1 * seed + c1) % mod;
	for (int i = 0; i < count; i++) {
		int1 = (a2 * int1 + c2) % mod;
		int2 = (a3 * int2 + c3) % mod;
	}
	const size_t start = blockIdx.x * N / gridDim.x;
	const size_t stop = (blockIdx.x + 1) * N / gridDim.x;
	for (size_t i = start + thread; i < stop; i += block_size) {
		int1 = (a4 * (uint64_t) int1 + c4) % mod;
		int2 = (a5 * (uint64_t) int2 + c5) % mod;
		const float x = ((float) int1 + 0.5f) / (float) uint64_t(mod + uint64_t(1));
		const float y1 = ((float) int2 + 0.5f) / (float) uint64_t(mod + uint64_t(1));
		const float y = 2.f * (float) M_PI * y1;
		nums[i] = sqrtf(-logf(fabsf(x))) * expc(cmplx(0, 1) * y);
	}
	__syncthreads();
}


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
