/*
 * zeldovich.hpp
 *
 *  Created on: Jan 12, 2021
 *      Author: dmarce1
 */

#include <cosmictiger/zeldovich.hpp>
#include <cosmictiger/array.hpp>
#include <curand_kernel.h>

void execute_2lpt_kernel(std::vector<cmplx>& Y, int xbegin, int xend, const interp_functor<float> den_k, int N,
		float box_size, int dim1, int dim2) {
	cudaFuncAttributes attribs;
	cuda_set_device();
	CUDA_CHECK(cudaFuncGetAttributes(&attribs, _2lpt_kernel));
	const int nthreads = std::min(attribs.maxThreadsPerBlock, N);
	cmplx* dev_ptr;
	CUDA_CHECK(cudaMalloc(&dev_ptr, sizeof(cmplx) * N * N));
	CHECK_POINTER(dev_ptr);
	for (int xi = xbegin; xi < xend; xi++) {
		CUDA_CHECK(cudaMemcpy(dev_ptr, Y.data() + (xi - xbegin) * N * N, sizeof(cmplx) * N * N, cudaMemcpyHostToDevice));
		_2lpt_kernel<<<1,nthreads>>>(dev_ptr,xi,den_k,N,box_size,dim1,dim2);
		CUDA_CHECK(cudaMemcpy(Y.data() + (xi - xbegin) * N * N, dev_ptr, sizeof(cmplx) * N * N, cudaMemcpyDeviceToHost));
	}
	CUDA_CHECK(cudaFree(dev_ptr));
}

void execute_2lpt_correction_kernel(std::vector<cmplx>& Y, int xbegin, int xend, int N, float box_size, int dim) {
	cudaFuncAttributes attribs;
	cuda_set_device();
	CUDA_CHECK(cudaFuncGetAttributes(&attribs, _2lpt_correction_kernel));
	const int nthreads = std::min(attribs.maxThreadsPerBlock, N);
	cmplx* dev_ptr;
	CUDA_CHECK(cudaMalloc(&dev_ptr, sizeof(cmplx) * N * N));
	CHECK_POINTER(dev_ptr);
	for (int xi = xbegin; xi < xend; xi++) {
		CUDA_CHECK(cudaMemcpy(dev_ptr, Y.data() + (xi - xbegin) * N * N, sizeof(cmplx) * N * N, cudaMemcpyHostToDevice));
		_2lpt_correction_kernel<<<1,nthreads>>>(dev_ptr,xi,xi+1,N,box_size,dim);
		CUDA_CHECK(cudaMemcpy(Y.data() + (xi - xbegin) * N * N, dev_ptr, sizeof(cmplx) * N * N, cudaMemcpyDeviceToHost));
	}
	CUDA_CHECK(cudaFree(dev_ptr));
}

__global__
void _2lpt_kernel(cmplx* Y, int xbegin, const interp_functor<float> den_k, int N, float box_size, int dim1, int dim2) {
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	const int bsz = blockDim.x;
	const float factor = powf(box_size, -1.5);
	int i = xbegin + bid;
	const unsigned sequence = abs(i) * bsz + tid;
	curandState_t rand;
	curand_init(1234, sequence, 0, &rand);
	int i0 = i < N / 2 ? i : i - N;
	float kx = 2.f * (float) M_PI / box_size * float(i0);
	for (int j = tid; j < N; j += bsz) {
		int j0 = j < N / 2 ? j : j - N;
		float ky = 2.f * (float) M_PI / box_size * float(j0);
		for (int l = 0; l < N; l++) {
			int l0 = l < N / 2 ? l : l - N;
			int i2 = i0 * i0 + j0 * j0 + l0 * l0;
			int index0;
			index0 = N * (N * (i - xbegin) + j) + l;
			if (i2 > 0 && i2 < N * N / 4) {
				float kz = 2.f * (float) M_PI / box_size * float(l0);
				float k2 = kx * kx + ky * ky + kz * kz;
				float k = sqrtf(kx * kx + ky * ky + kz * kz);
				const cmplx K[NDIM + 1] = { { kx, 0 }, { ky, 0 }, { kz, 0 }, { 0, -1 } };
				const cmplx spec = sqrtf(den_k(k)) * factor * K[dim1] * K[dim2] / k2;
				const float x = (float(curand(&rand)) + 0.5f) / (float(0xFFFFFFFFUL) + 1.f);
				const float y = (float(curand(&rand)) + 0.5f) / (float(0xFFFFFFFFUL) + 1.f);
				Y[index0] = spec * sqrtf(-logf(fabsf(x))) * expc(cmplx(0, 1) * 2.f * float(M_PI) * y);
			} else {
				Y[index0].real() = Y[index0].imag() = 0.0;
			}
		}
	}
}

__global__
void _2lpt_correction_kernel(cmplx* Y, int xbegin, int xend, int N, float box_size, int dim) {
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	const int bsz = blockDim.x;
	const int gsz = gridDim.x;
	const float factor = powf(box_size, -1.5);
	for (int i = xbegin + bid; i < xend; i += gsz) {
		int i0 = i < N / 2 ? i : i - N;
		float kx = 2.f * (float) M_PI / box_size * float(i0);
		for (int j = tid; j < N; j += bsz) {
			int j0 = j < N / 2 ? j : j - N;
			float ky = 2.f * (float) M_PI / box_size * float(j0);
			for (int l = 0; l < N; l++) {
				int l0 = l < N / 2 ? l : l - N;
				int i2 = i0 * i0 + j0 * j0 + l0 * l0;
				int index0 = N * (N * (i - xbegin) + j) + l;
				if (i2 > 0) {
					float kz = 2.f * (float) M_PI / box_size * float(l0);
					float k2 = kx * kx + ky * ky + kz * kz;
					const float K[NDIM + 1] = { kx, ky, kz };
					Y[index0] = -cmplx(0, 1) * K[dim] * Y[index0] / k2;
				} else {
					Y[index0] = cmplx(0, 0);
				}
			}
		}
	}
}
