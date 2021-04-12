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
		if (iters > 100000) {
			printf("Finished early with error = %e\n", err);
			break;
		}
	} while (err > 1.0e-12);
	return x;
}

__global__
void generate_random_normals(cmplx* nums, int N, int seed) {
	curandState_t state;
	const int& thread = threadIdx.x;
	curand_init(seed, thread, 0, &state);

	for (int i = thread; i < N; i += blockDim.x) {
		float x1 = curand_uniform(&state);
		float y1 = curand_uniform(&state);
		float x = x1;
		float y = 2.f * (float) M_PI * y1;
//		printf( "%i %i\n", i, N);
		nums[i] = sqrtf(-logf(fabsf(x))) * expc(cmplx(0, 1) * y);
	}
	__syncthreads();
}
