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
	for (auto i = start + tid; i < stop; i += bsz) {
		const auto x = parts.pos(0, i).to_double();
		const auto y = parts.pos(1, i).to_double();
		const auto z = parts.pos(2, i).to_double();
		const auto xi = ((int) (x * (double) N)) % N;
		const auto yi = ((int) (y * (double) N)) % N;
		const auto zi = ((int) (z * (double) N)) % N;
		const auto index = (xi * N + yi) * N + zi;
		atomicAdd(&(den_k[index].real()), mass);
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
				atomicAdd(spec + index, den_k[index].norm());
				atomicAdd(count + index, 1);
			}
		}
	}
}

void compute_power_spectrum(particle_set& parts, int filenum) {
	cmplx* den;
	size_t N = global().opts.parts_dim;
	size_t N3 = N * sqr(N);
	float* spec;
	int* count;

	printf("%e GB\n", cuda_unified_total() / 1024.0 / 1024 / 1024);
	CUDA_MALLOC(den, N3);
	printf("%e GB\n", cuda_unified_total() / 1024.0 / 1024 / 1024);
	CUDA_MALLOC(spec, N / 2);
	printf("%e GB\n", cuda_unified_total() / 1024.0 / 1024 / 1024);
	CUDA_MALLOC(count, N / 2);
	printf("%e GB\n", cuda_unified_total() / 1024.0 / 1024 / 1024);

	for (int i = 0; i < N3; i++) {
		den[i] = -1.0;
	}
	for (int i = 0; i < N / 2; i++) {
		spec[i] = 0.f;
		count[i] = 0;
	}
	cudaFuncAttributes attrib;
	CUDA_CHECK(cudaFuncGetAttributes(&attrib, power_spectrum_init));
	int block_size = attrib.maxThreadsPerBlock;
	int num_blocks;
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, power_spectrum_init, block_size, 0));
	num_blocks *= global().cuda.devices[0].multiProcessorCount;
	power_spectrum_init<<<num_blocks,block_size>>>(parts.get_virtual_particle_set(),den,N,(double) N3 / (double)parts.size());
	CUDA_CHECK(cudaDeviceSynchronize());
	fft3d(den, N);
	CUDA_CHECK(cudaFuncGetAttributes(&attrib, power_spectrum_compute));
	block_size = std::min((int) N, attrib.maxThreadsPerBlock);
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, power_spectrum_compute, block_size, 0));
	num_blocks *= global().cuda.devices[0].multiProcessorCount;
	power_spectrum_compute<<<num_blocks,block_size>>>(den,N, spec,count);
	CUDA_CHECK(cudaDeviceSynchronize());
	std::string filename = std::string("power.") + std::to_string(filenum) + std::string(".txt");
	const auto code_to_mpc = global().opts.code_to_cm / constants::mpc_to_cm;
	for (int i = 0; i < N / 2; i++) {
		spec[i] /= count[i];
		spec[i] *= std::pow(code_to_mpc, 3) / (8 * N3 * N3);
	}
	FILE* fp = fopen(filename.c_str(), "wt");
	if (fp == NULL) {
		printf("Unable to open %s for writing\n", filename.c_str());
		abort();
	}
	for (int i = 0; i < N / 2; i++) {
		const auto k = 2.0 * M_PI * (i + 0.5) / code_to_mpc;
		fprintf(fp, "%e %e\n", k, spec[i]);
	}
	fclose(fp);
	CUDA_FREE(den);
	CUDA_FREE(spec);
	CUDA_FREE(count);
}

