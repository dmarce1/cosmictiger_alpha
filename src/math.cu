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
void generate_random_vectors(fixed32* x, fixed32* y, fixed32* z, size_t N, int seed) {
	const int& thread = threadIdx.x;
	const int& block_size = blockDim.x;
	const auto sequence = threadIdx.x + blockIdx.x * blockDim.x;
	curandState_t rand;
	curand_init(1234, sequence, 0, &rand);
	const size_t start = blockIdx.x * N / gridDim.x;
	const size_t stop = (blockIdx.x + 1) * N / gridDim.x;
	for (size_t i = start + thread; i < stop; i += block_size) {
		*((unsigned int*) x + i) = curand(&rand);
		*((unsigned int*) y + i) = curand(&rand);
		*((unsigned int*) z + i) = curand(&rand);
	}
	__syncthreads();
}
