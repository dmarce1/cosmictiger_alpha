#include <cosmictiger/fourier.hpp>
#include <cosmictiger/global.hpp>
#include <cufft.h>

#define FFTSIZE_COMPUTE 32
#define FFTSIZE_TRANSPOSE 32

/* error checker from https://forums.developer.nvidia.com/t/cufft-error-handling/29231 */
static const char *_cudaGetErrorEnum(cufftResult error) {
	switch (error) {
	case CUFFT_SUCCESS:
		return "CUFFT_SUCCESS";

	case CUFFT_INVALID_PLAN:
		return "CUFFT_INVALID_PLAN";

	case CUFFT_ALLOC_FAILED:
		return "CUFFT_ALLOC_FAILED";

	case CUFFT_INVALID_TYPE:
		return "CUFFT_INVALID_TYPE";

	case CUFFT_INVALID_VALUE:
		return "CUFFT_INVALID_VALUE";

	case CUFFT_INTERNAL_ERROR:
		return "CUFFT_INTERNAL_ERROR";

	case CUFFT_EXEC_FAILED:
		return "CUFFT_EXEC_FAILED";

	case CUFFT_SETUP_FAILED:
		return "CUFFT_SETUP_FAILED";

	case CUFFT_INVALID_SIZE:
		return "CUFFT_INVALID_SIZE";

	case CUFFT_UNALIGNED_DATA:
		return "CUFFT_UNALIGNED_DATA";
	}

	return "<unknown>";
}

inline void _cuda_fft_check(cufftResult err, const char *file, const int line) {
	if (CUFFT_SUCCESS != err) {
		fprintf(stderr, "CUFFT error in file '%s', line %d\n %s\nerror %d: %s\nterminating!\n", __FILE__, __LINE__, err,
				_cudaGetErrorEnum(err));
		cudaDeviceReset();
		assert(0);
	}
}

#define CUDA_FFT_CHECK(a) _cuda_fft_check(a,__FILE__,__LINE__)

__global__
void transpose_2d(cmplx* Y, int N) {
	const int& tid = threadIdx.x;
	const int block_size = blockDim.x;
	const int& bid = blockIdx.x;
	const int& grid_size = gridDim.x;
	for (int xi = bid; xi < N; xi += grid_size) {
		for (int yi = tid; yi < N; yi += block_size) {
			if (xi < yi) {
				const int i1 = (N * xi + yi);
				const int i2 = (N * yi + xi);
				const cmplx tmp = Y[i1];
				Y[i1] = Y[i2];
				Y[i2] = tmp;
			}
		}
	}
}

__global__
void normalize_invert_2d(cmplx* Y, int N) {
	const int& tid = threadIdx.x;
	const int block_size = blockDim.x;
	const int& bid = blockIdx.x;
	const int& grid_size = gridDim.x;
	const float N3inv = 1.0f / (N * sqr(N));
	for (int xi = bid; xi < N; xi += grid_size) {
		for (int yi = tid; yi < N; yi += block_size) {
			const int i1 = (N * xi + yi);
			Y[i1].real() *= N3inv;
			Y[i1].imag() *= N3inv;
		}
	}

}
__global__
void transpose_xy_3d(cmplx* Y, int N) {
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
void normalize_invert_3d(cmplx* Y, int N) {
	const int& tid = threadIdx.x;
	const int block_size = blockDim.x;
	const int& bid = blockIdx.x;
	const int& grid_size = gridDim.x;
	const float N3inv = 1.0f / (N * sqr(N));
	for (int xy = bid; xy < N * N; xy += grid_size) {
		int xi = xy / N;
		int yi = xy % N;
		for (int zi = tid; zi < N; zi += block_size) {
			const int i1 = N * (N * xi + yi) + zi;
			Y[i1].real() *= N3inv;
			Y[i1].imag() *= N3inv;
		}
	}

}

__global__
void transpose_xz_3d(cmplx* Y, int N) {
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
void transpose_yz_3d(cmplx* Y, int N) {
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
	cuda_set_device();
	const int maxgrid = global().cuda.devices[0].maxGridSize[0];
	int nblocksc = min(N * N * N / FFTSIZE_COMPUTE, maxgrid);
	int nblockst = min(N * N * N / FFTSIZE_TRANSPOSE, maxgrid);
	cufftHandle plan;
	CUDA_FFT_CHECK(cufftPlan1d(&plan, N, CUFFT_C2C, N * N));
	CUDA_FFT_CHECK(cufftExecC2C(plan, (cufftComplex *)Y, (cufftComplex *)Y, CUFFT_FORWARD));
	transpose_yz_3d<<<nblockst,FFTSIZE_TRANSPOSE>>>(Y,N);
	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_FFT_CHECK(cufftExecC2C(plan, (cufftComplex *)Y, (cufftComplex *)Y, CUFFT_FORWARD));
	transpose_xz_3d<<<nblockst,FFTSIZE_TRANSPOSE>>>(Y,N);
	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_FFT_CHECK(cufftExecC2C(plan, (cufftComplex *)Y, (cufftComplex *)Y, CUFFT_FORWARD));
	transpose_yz_3d<<<nblockst,FFTSIZE_TRANSPOSE>>>(Y,N);
	transpose_xy_3d<<<nblockst,FFTSIZE_TRANSPOSE>>>(Y,N);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void fft3d_inv(cmplx* Y, int N) {
	cuda_set_device();
	const int maxgrid = global().cuda.devices[0].maxGridSize[0];
	int nblocks = min(N * N * N / 32, maxgrid);
	normalize_invert_3d<<<nblocks,32>>>(Y,N);
	fft3d(Y, N);
}

void fft2d(cmplx* Y, int N) {
	cuda_set_device();
	const int maxgrid = global().cuda.devices[0].maxGridSize[0];
	int nblocksc = min(N * N / FFTSIZE_COMPUTE, maxgrid);
	int nblockst = min(N * N / FFTSIZE_TRANSPOSE, maxgrid);
	cufftHandle plan;
	CUDA_FFT_CHECK(cufftPlan1d(&plan, N, CUFFT_C2C, N));
	CUDA_FFT_CHECK(cufftExecC2C(plan, (cufftComplex *)Y, (cufftComplex *)Y, CUFFT_FORWARD));
	transpose_2d<<<nblockst,FFTSIZE_TRANSPOSE>>>(Y,N);
	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_FFT_CHECK(cufftPlan1d(&plan, N, CUFFT_C2C, N));
	transpose_2d<<<nblockst,FFTSIZE_TRANSPOSE>>>(Y,N);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void fft1d(cmplx* Y, int N) {
	cuda_set_device();
	const int maxgrid = global().cuda.devices[0].maxGridSize[0];
	int nblocksc = min(N * N / FFTSIZE_COMPUTE, maxgrid);
	int nblockst = min(N * N / FFTSIZE_TRANSPOSE, maxgrid);
	cufftHandle plan;
	CUDA_FFT_CHECK(cufftPlan1d(&plan, N, CUFFT_C2C, N));
	CUDA_FFT_CHECK(cufftExecC2C(plan, (cufftComplex *)Y, (cufftComplex *)Y, CUFFT_FORWARD));
	CUDA_CHECK(cudaDeviceSynchronize());
}

void fft32_inv(cmplx* Y, int N) {
	cuda_set_device();
	const int maxgrid = global().cuda.devices[0].maxGridSize[0];
	int nblocks = min(N * N / 32, maxgrid);
	normalize_invert_2d<<<nblocks,32>>>(Y,N);
	fft2d(Y, N);
}
