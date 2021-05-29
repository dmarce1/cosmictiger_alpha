#include <cosmictiger/power.hpp>
#include <cosmictiger/fourier.hpp>
#include <cosmictiger/particle.hpp>
#include <cosmictiger/constants.hpp>
#include <cosmictiger/global.hpp>

__global__ void power_spectrum_init(particle_set parts, cmplx* den_k, size_t N, float mass0) {
	const auto& tid = threadIdx.x;
	const auto& bid = blockIdx.x;
	const auto& bsz = blockDim.x;
	const auto& gsz = gridDim.x;
	const auto start = (size_t) bid * (size_t) parts.size() / (size_t) gsz;
	const auto stop = (size_t) (bid + 1) * (size_t) parts.size() / (size_t) gsz;
	const float floatN = (float) N;
	const float mass = mass0;
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
		const auto i0 = i < N / 2 ? i : i - N;
		const auto j0 = j < N / 2 ? j : j - N;
		for (int k = tid; k < N; k += bsz) {
			const auto k0 = k < N / 2 ? k : k - N;
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
		PRINT("Unable to open %s for writing\n", filename.c_str());
		abort();
	}
	const auto code_to_mpc = global().opts.code_to_cm / constants::mpc_to_cm;
	for (int i = 1; i < N / 2; i++) {
		const auto k = 2.0 * M_PI * (i + 0.5) / code_to_mpc;
		fprintf(fp, "%e %e\n", k, spec[i]);
	}
	fclose(fp);
	CUDA_FREE(den);
	CUDA_FREE(spec);
}

std::function<float(float)> eisenstein_and_hu(float omega_b, float omega_c, float Theta, float h, float ns) {
	std::function<float(float)> P;
	float omega0 = omega_b + omega_c;
	float omega0h2 = omega0 * sqr(h);
	float omegabh2 = omega_b * sqr(h);
	float zeq = float(2.5e4) * omega0h2 * powf(Theta, -4.f);
	float keq = float(7.46e-2) * omega0h2 * powf(Theta, -2.f);
	float b1 = 0.313f * powf(omega0h2, -0.419f) * (1.f + 0.607f * powf(omega0h2, 0.674f));
	float b2 = 0.238f * powf(omega0h2, 0.223f);
	float zd = 1291.f * powf(omega0h2, 0.251f) / (1.f + 0.659f * powf(omega0h2, 0.828f))
			* (1.0f + b1 * powf(omegabh2, b2));
	const auto R = [omegabh2,Theta](float z) {
		return 31.5f * omegabh2 * powf(Theta,-4.0)*(1000.0f/z);
	};
	float Req = R(zeq);
	float Rd = R(zd);
	float s = (2.0f / (3.0 * keq)) * sqrtf(6.0f / Req) * logf((sqrtf(1.f + Rd) + sqrtf(Rd + Req)) / (1.0f + sqrtf(Req)));
	float ksilk = 1.6f * powf(omegabh2, 0.52f) * powf(omega0h2, 0.73f) * (1.f + powf(10.4f * omega0h2, -0.95f));
	const auto q = [keq](float k) {
		return k / (13.41f * keq);
	};
	float a1 = powf(46.9f * omega0h2, 0.670f) * (1.f + powf(32.1f * omega0h2, -0.532f));
	float a2 = powf(12.0f * omega0h2, 0.424f) * (1.f + powf(45.0f * omega0h2, -0.582f));
	float alpha_c = powf(a1, -omega_b / omega0) * powf(a2, -pow(omega_b / omega0, 3.f));
	float b12 = 0.944f / (1.f + powf(458.f * omega0h2, -0.708f));
	float b22 = powf(0.395f * omega0h2, -.0266f);
	float beta_c = 1.0f / (1.0f + b12 * (powf(omega_c / omega0, b22) - 1.f));
	const auto G = [](float y) {
		return y * (-6.f * sqrtf(1.f+y)+(2.f+3.f*y)*logf((sqrtf(1.f+y)+1.f)/(sqrtf(1.f+y)-1.f)));
	};
	float alpha_b = 2.07f * keq * s * powf(1.0f + Rd, -0.75f) * G((1.f + zeq) / (1.f + zd));
	const auto T0 = [q](float k, float alpha_c, float beta_c) {
		const float e = expf(1.0f);
		const float C = 14.2f / alpha_c + 386.f / (1.0f + 69.9f * powf(q(k),1.08f));
		const float tmp = logf(e+1.8f*beta_c*q(k));
		return tmp / (tmp+C*sqr(q(k)));
	};
	const auto Tc = [beta_c,alpha_c,T0,s](float k) {
		float f = 1.0f / (1.0f + powf(k * s / 5.4f, 4.0f));
		return f * T0(k,1.0,beta_c) + (1.0f-f)*T0(k,alpha_c,beta_c);
	};
	float beta_b = 0.5f + omega_b / omega0 + (3.f - 2.f * omega_b / omega0) * sqrtf(sqr(17.2f * omega0h2) + 1.f);
	float beta_node = 8.41f * powf(omega0h2, 0.435f);
	const auto Tb =
			[s,T0,beta_b,ksilk, alpha_b,beta_node](float k) {
				float s_tilde = s / powf(1.f + powf(beta_node / (k * s), 3.f), 1.f / 3.f);
				return T0(k,1.f,1.f) / (1.0f +sqr(k*s/5.2f)) + alpha_b / (1.0f + powf(beta_b /(k*s),3.f)* expf(-powf(k/ksilk,1.4f))) * sinf(k*s_tilde) / (k*s_tilde);
			};
	return [Tc, Tb, ns, omega_b, omega_c, omega0](float k) {
		return powf(k,ns) * sqr((omega_c*Tc(k)+omega_b*Tb(k))/omega0);
	};
}
