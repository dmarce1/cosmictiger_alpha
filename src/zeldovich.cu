/*
 * zeldovich.hpp
 *
 *  Created on: Jan 12, 2021
 *      Author: dmarce1
 */

#include <cosmictiger/zeldovich.hpp>
#include <cosmictiger/array.hpp>

void execute_2lpt_kernel(cmplx* Y, int xbegin, int xend, const interp_functor<float> den_k, int N, float box_size, int dim1,
		int dim2) {
	cudaFuncAttributes attribs;
	CUDA_CHECK(cudaFuncGetAttributes(&attribs, _2lpt_kernel));
	const int nthreads = std::min(attribs.maxThreadsPerBlock, N);
	const int nblocks = xend - xbegin;
	_2lpt_kernel<<<nblocks,nthreads>>>(Y,xbegin,xend,den_k,N,box_size,dim1,dim2);
	CUDA_CHECK(cudaDeviceSynchronize());
}


__global__
void _2lpt_kernel(cmplx* Y, int xbegin, int xend, const interp_functor<float> den_k, int N, float box_size, int dim1,
		int dim2) {
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
				if (i2 > 0 && i2 < N * N / 4) {
					float kz = 2.f * (float) M_PI / box_size * float(l0);
					float k2 = kx * kx + ky * ky + kz * kz;
					float k = sqrtf(kx * kx + ky * ky + kz * kz);
					const cmplx K[NDIM + 1] = { { kx, 0 }, { ky, 0 }, { kz, 0 }, { 0, -1 } };
					const cmplx number = sqrtf(den_k(k)) * factor * K[dim1] * K[dim2] / k2;
					Y[index0] = Y[index0] * number;
				} else {
					Y[index0].real() = Y[index0].imag() = 0.0;
				}
			}
		}
	}
}

__global__
void zeldovich(cmplx* phi, const cmplx* rands, const interp_functor<float>* Pptr, float box_size, int N, int dim,
		zeldovich_t type) {
	const int& thread = threadIdx.x;
	const int& block_size = blockDim.x;
	auto& P = *Pptr;
	for (int i = thread; i < N * N * N; i += block_size) {
		phi[i] = cmplx(0, 0);
	}
	__syncthreads();
	for (int ij = thread; ij < N * N; ij += block_size) {
		int i = ij / N;
		int j = ij % N;
		int i0 = i < N / 2 ? i : i - N;
		int j0 = j < N / 2 ? j : j - N;
		float kx = 2.f * (float) M_PI / box_size * float(i0);
		float ky = 2.f * (float) M_PI / box_size * float(j0);
		for (int l = 0; l < N / 2; l++) {
			int l0 = l < N / 2 ? l : l - N;
			int i2 = i0 * i0 + j0 * j0 + l0 * l0;
			int index0 = N * (N * i + j) + l;
			int index1;
			index1 = N * (N * ((N - i) % N) + ((N - j) % N)) + ((N - l) % N);
			if (i2 > 0 && i2 < N * N / 4) {
				float kz = 2.f * (float) M_PI / box_size * float(l0);
				float K[NDIM] = { kx, ky, kz };
				float k = sqrt(kx * kx + ky * ky + kz * kz);
				switch (type) {
				case VELOCITY:
					phi[index0] = -cmplx(0, 1) * (rands[index0] * sqrtf(P(k))) * K[dim] / k * powf(box_size, -1.5);
					break;
				case DISPLACEMENT:
					phi[index0] = -cmplx(0, 1) * (rands[index0] * sqrtf(P(k))) * K[dim] / (k * k) * powf(box_size, -1.5);
					break;
				case DENSITY:
					phi[index0] = rands[index0] * sqrtf(P(k)) * powf(box_size, -1.5);
				}
				phi[index1] = phi[index0].conj();
			}
		}
	}
	__syncthreads();
	/*	if (thread == 0) {
	 fft3d(phi, basis, N);
	 }
	 __syncthreads();
	 for (int i = thread; i < N * N * N; i += block_size) {
	 maxdisp[thread] = max(maxdisp[thread], abs(phi[i].real()));
	 }
	 __syncthreads();
	 for (int P = block_size / 2; P >= 1; P /= 2) {
	 if (thread < P) {
	 maxdisp[thread] = max(maxdisp[thread], maxdisp[thread + P]);
	 }
	 __syncthreads();
	 }
	 *res = maxdisp[0];*/
}

