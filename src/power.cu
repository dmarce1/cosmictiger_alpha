#include <cosmictiger/power.hpp>
#include <cosmictiger/fourier.hpp>
#include <cosmictiger/particle.hpp>
#include <cosmictiger/constants.hpp>
#include <cosmictiger/global.hpp>

__global__ void power_spectrum_init(particle_set parts, cmplx* den_k, size_t N, float mass) {
	const auto& tid = threadIdx.x;
	const auto& bid = blockIdx.x;
	const auto& bsz = blockDim.x;
	const auto& gsz = gridDim.x;
	const auto start = bid * parts.size() / gsz;
	const auto stop = (bid + 1) * parts.size() / gsz;
	const float floatN = (float) N;
	for (auto i = start + tid; i < stop; i += bsz) {
		const auto x = parts.pos(0, i).to_float() * floatN;
		const auto y = parts.pos(1, i).to_float() * floatN;
		const auto z = parts.pos(2, i).to_float() * floatN;
		const int xi0 = (int) x % N;
		const int yi0 = (int) y % N;
		const int zi0 = (int) z % N;
		const int xi1 = (xi0 + 1) % N;
		const int yi1 = (yi0 + 1) % N;
		const int zi1 = (zi0 + 1) % N;
		const float w1x = x - xi0;
		const float w1y = y - yi0;
		const float w1z = z - zi0;
		const float w0x = 1.f - w1x;
		const float w0y = 1.f - w1y;
		const float w0z = 1.f - w1z;
		atomicAdd(&(den_k[xi0 * N * N + yi0 * N + zi0].real()), mass * w0x * w0y * w0z);
		atomicAdd(&(den_k[xi0 * N * N + yi0 * N + zi1].real()), mass * w0x * w0y * w1z);
		atomicAdd(&(den_k[xi0 * N * N + yi1 * N + zi0].real()), mass * w0x * w1y * w0z);
		atomicAdd(&(den_k[xi0 * N * N + yi1 * N + zi1].real()), mass * w0x * w1y * w1z);
		atomicAdd(&(den_k[xi1 * N * N + yi0 * N + zi0].real()), mass * w1x * w0y * w0z);
		atomicAdd(&(den_k[xi1 * N * N + yi0 * N + zi1].real()), mass * w1x * w0y * w1z);
		atomicAdd(&(den_k[xi1 * N * N + yi1 * N + zi0].real()), mass * w1x * w1y * w0z);
		atomicAdd(&(den_k[xi1 * N * N + yi1 * N + zi1].real()), mass * w1x * w1y * w1z);
	}
}

__global__ void power_spectrum_compute(cmplx* den_k, size_t N, float* spec, int* count) {
	const auto& tid = threadIdx.x;
	const auto& bid = blockIdx.x;
	const auto& bsz = blockDim.x;
	const auto& gsz = gridDim.x;
	const int cutoff = N / 2;
	for (int ij = bid; ij < N * N; ij += gsz) {
		int i = ij / N;
		int j = ij % N;
		const auto i0 = i < N / 2 ? i : N - i;
		const auto j0 = j < N / 2 ? j : N - j;
		for (int k = tid; k < N; k += bsz) {
			const auto k0 = k < N / 2 ? k : N - k;
			const auto wavenum2 = i0 * i0 + j0 * j0 + k0 * k0;
			const auto wavenum = sqrtf((float) wavenum2);
			int index = (int) (wavenum);
			if (index < cutoff) {
				atomicAdd(spec + index, den_k[i * N * N + j * N + k].norm());
				atomicAdd(count + index, 1);
			}
		}
	}
}

void compute_power_spectrum(cmplx* den, float* spec, int N) {
	size_t N3 = N * sqr(N);
	int* count;

	CUDA_MALLOC(count, N / 2);

	for (int i = 0; i < N / 2; i++) {
		spec[i] = 0.f;
		count[i] = 0;
	}
	fft3d(den, N);
	cudaFuncAttributes attrib;
	CUDA_CHECK(cudaFuncGetAttributes(&attrib, power_spectrum_compute));
	int num_blocks;
	int block_size = std::min((int) N, attrib.maxThreadsPerBlock);
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, power_spectrum_compute, block_size, 0));
	num_blocks *= global().cuda.devices[0].multiProcessorCount;
	power_spectrum_compute<<<num_blocks,block_size>>>(den,N, spec,count);
	CUDA_CHECK(cudaDeviceSynchronize());
	const auto code_to_mpc = global().opts.code_to_cm / constants::mpc_to_cm;
	for (int i = 0; i < N / 2; i++) {
		spec[i] /= count[i];
		spec[i] *= std::pow(code_to_mpc, 3) / (N3 * N3);
	}
	CUDA_FREE(count);
}

void compute_particle_power_spectrum(particle_set& parts, int filenum) {
	cmplx* den;
	size_t N = global().opts.parts_dim;
	size_t N3 = N * sqr(N);
	float* spec;

	CUDA_MALLOC(den, N3);
	CUDA_MALLOC(spec, N / 2);

	for (int i = 0; i < N3; i++) {
		den[i] = -1.0;
	}
	cudaFuncAttributes attrib;
	CUDA_CHECK(cudaFuncGetAttributes(&attrib, power_spectrum_init));
	int block_size = attrib.maxThreadsPerBlock;
	int num_blocks;
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, power_spectrum_init, block_size, 0));
	num_blocks *= global().cuda.devices[0].multiProcessorCount;
	power_spectrum_init<<<num_blocks,block_size>>>(parts.get_virtual_particle_set(),den,N,(float) N3 / (float)parts.size());
	CUDA_CHECK(cudaDeviceSynchronize());
	compute_power_spectrum(den, spec, N);
	std::string filename = std::string("power.") + std::to_string(filenum) + std::string(".txt");
	FILE* fp = fopen(filename.c_str(), "wt");
	if (fp == NULL) {
		printf("Unable to open %s for writing\n", filename.c_str());
		abort();
	}
	const auto code_to_mpc = global().opts.code_to_cm / constants::mpc_to_cm;
	for (int i = 1; i < N / 2; i++) {
		const auto k = 2.0 * M_PI * (i+0.5) / code_to_mpc;
		fprintf(fp, "%e %e\n", k, spec[i]);
	}
	fclose(fp);
	CUDA_FREE(den);
	CUDA_FREE(spec);
}

