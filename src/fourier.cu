#include <cosmictiger/fourier.hpp>
#include <cosmictiger/global.hpp>

#define FFTSIZE_COMPUTE 32
#define FFTSIZE_TRANSPOSE 32

__global__
void fft(cmplx* Y, const cmplx* expi, int N) {
	const int& tid = threadIdx.x;
	const int block_size = blockDim.x;
	const int& bid = blockIdx.x;
	const int& grid_size = gridDim.x;
	int level = 0;
	for (int i = N; i > 1; i >>= 1) {
		level++;
	}
	for (int xy = bid; xy < N * N; xy += grid_size) {
		int xi = xy / N;
		int yi = xy % N;
		int offset = N * (N * xi + yi);
		cmplx* y = Y + offset;

		for (int i = tid; i < N; i += block_size) {
			int j = 0;
			int l = i;
			for (int k = 0; k < level; k++) {
				j = (j << 1) | (l & 1);
				l >>= 1;
			}
			if (j > i) {
				auto tmp = y[i];
				y[i] = y[j];
				y[j] = tmp;
			}
		}
		__syncthreads();
		for (int P = 2; P <= N; P *= 2) {
			const int s = N / P;
			if (N > P * P) {
				for (int i = P * tid; i < N; i += P * block_size) {
					int k = 0;
					for (int j = i; j < i + P / 2; j++) {
						const auto t = y[j + P / 2] * expi[k];
						y[j + P / 2] = y[j] - t;
						y[j] += t;
						k += s;
					}
				}
				__syncthreads();
			} else {
				for (int i = 0; i < N; i += P) {
					int k = s * tid;
					for (int j = i + tid; j < i + P / 2; j += block_size) {
						const auto t = y[j + P / 2] * expi[k];
						y[j + P / 2] = y[j] - t;
						y[j] += t;
						k += s * block_size;
					}
					__syncthreads();
				}
			}
		}
	}

}

__global__
void transpose_xy(cmplx* Y, int N) {
	const int& tid = threadIdx.x;
	const int block_size = blockDim.x;
	const int& bid = blockIdx.x;
	const int& grid_size = gridDim.x;

	for (int xy = bid; xy < N * N; xy += grid_size) {
		int xi = xy / N;
		int yi = xy % N;
		if (xi < yi) {
			for (int zi = tid; zi < N; zi += block_size) {
				const int i1 = N * (N * xi + yi) + zi;
				const int i2 = N * (N * yi + xi) + zi;
				const cmplx tmp = Y[i1];
				Y[i1] = Y[i2];
				Y[i2] = tmp;
			}
		}
	}

}

__global__
void transpose_xz(cmplx* Y, int N) {
	const int& tid = threadIdx.x;
	const int block_size = blockDim.x;
	const int& bid = blockIdx.x;
	const int& grid_size = gridDim.x;
	for (int xy = bid; xy < N * N; xy += grid_size) {
		int xi = xy / N;
		int yi = xy % N;
		for (int zi = tid; zi < xi; zi += block_size) {
			const int i1 = N * (N * xi + yi) + zi;
			const int i2 = N * (N * zi + yi) + xi;
			const cmplx tmp = Y[i1];
			Y[i1] = Y[i2];
			Y[i2] = tmp;
		}
	}
}

__global__
void transpose_yz(cmplx* Y, int N) {
	const int& tid = threadIdx.x;
	const int block_size = blockDim.x;
	const int& bid = blockIdx.x;
	const int& grid_size = gridDim.x;
	for (int xy = bid; xy < N * N; xy += grid_size) {
		int xi = xy / N;
		int yi = xy % N;
		for (int zi = tid; zi < yi; zi += block_size) {
			const int i1 = N * (N * xi + yi) + zi;
			const int i2 = N * (N * xi + zi) + yi;
			const cmplx tmp = Y[i1];
			Y[i1] = Y[i2];
			Y[i2] = tmp;
		}
	}
}

void fft3d(cmplx* Y, int N) {
	cmplx* expi;
	CUDA_MALLOC(expi, N / 2);
	const int maxgrid = global().cuda.devices[0].maxGridSize[0];
	int nblocksc = min(N * N * N / FFTSIZE_COMPUTE, maxgrid);
	int nblockst = min(N * N * N / FFTSIZE_TRANSPOSE, maxgrid);
	for (int i = 0; i < N / 2; i++) {
		float omega = 2.0f * (float) M_PI * (float) i / (float) N;
		expi[i] = expc(-cmplx(0, 1) * omega);
	}
	fft<<<nblocksc,FFTSIZE_COMPUTE>>>(Y,expi,N);
	CUDA_CHECK(cudaDeviceSynchronize());
	transpose_yz<<<nblockst,FFTSIZE_TRANSPOSE>>>(Y,N);
	CUDA_CHECK(cudaDeviceSynchronize());
	fft<<<nblocksc,FFTSIZE_COMPUTE>>>(Y,expi,N);
	CUDA_CHECK(cudaDeviceSynchronize());
	transpose_xz<<<nblockst,FFTSIZE_TRANSPOSE>>>(Y,N);
	CUDA_CHECK(cudaDeviceSynchronize());
	fft<<<nblocksc,FFTSIZE_COMPUTE>>>(Y,expi,N);
	CUDA_CHECK(cudaDeviceSynchronize());
	transpose_yz<<<nblockst,FFTSIZE_TRANSPOSE>>>(Y,N);
	CUDA_CHECK(cudaDeviceSynchronize());
	transpose_xy<<<nblockst,FFTSIZE_TRANSPOSE>>>(Y,N);
	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_FREE(expi);
}
