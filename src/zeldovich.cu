/*
 * zeldovich.hpp
 *
 *  Created on: Jan 12, 2021
 *      Author: dmarce1
 */

#include <cosmictiger/zeldovich.hpp>
#include <cosmictiger/array.hpp>

__global__
void zeldovich(cmplx* phi, const cmplx* basis, const cmplx* rands, const interp_functor<float>* Pptr, float box_size,
		int N, int dim, zeldovich_t type) {
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

