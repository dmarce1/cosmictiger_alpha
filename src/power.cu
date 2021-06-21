#include <cosmictiger/power.hpp>
#include <cosmictiger/particle_server.hpp>
#include <cosmictiger/constants.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/kernel.hpp>

__global__ void matter_power_spectrum_kernel(particle_set parts, float*den_k, int xb, int xe, int yb, int ye, int zb,
		int ze, int N) {
	const auto& tid = threadIdx.x;
	const auto& bid = blockIdx.x;
	const auto& bsz = blockDim.x;
	const auto& gsz = gridDim.x;
	int yspan = ye - yb;
	int zspan = ze - zb;
	const auto start = (size_t) bid * (size_t) parts.size() / (size_t) gsz;
	const auto stop = (size_t) (bid + 1) * (size_t) parts.size() / (size_t) gsz;
	const float floatN = (float) N;
	for (auto i = start + tid; i < stop; i += bsz) {
		const auto x = parts.pos(0, i).to_float() * floatN;
		const auto y = parts.pos(1, i).to_float() * floatN;
		const auto z = parts.pos(2, i).to_float() * floatN;
		const int xi0 = (int) x;
		const int yi0 = (int) y;
		const int zi0 = (int) z;
		const int xi1 = (xi0 + 1);
		const int yi1 = (yi0 + 1);
		const int zi1 = (zi0 + 1);
		const float w1x = x - xi0;
		const float w1y = y - yi0;
		const float w1z = z - zi0;
		const float w0x = 1.f - w1x;
		const float w0y = 1.f - w1y;
		const float w0z = 1.f - w1z;
		atomicAdd(&(den_k[(xi0 - xb) * yspan * zspan + (yi0 - yb) * zspan + (zi0 - zb)]), w0x * w0y * w0z);
		atomicAdd(&(den_k[(xi0 - xb) * yspan * zspan + (yi0 - yb) * zspan + (zi1 - zb)]), w0x * w0y * w1z);
		atomicAdd(&(den_k[(xi0 - xb) * yspan * zspan + (yi1 - yb) * zspan + (zi0 - zb)]), w0x * w1y * w0z);
		atomicAdd(&(den_k[(xi0 - xb) * yspan * zspan + (yi1 - yb) * zspan + (zi1 - zb)]), w0x * w1y * w1z);
		atomicAdd(&(den_k[(xi1 - xb) * yspan * zspan + (yi0 - yb) * zspan + (zi0 - zb)]), w1x * w0y * w0z);
		atomicAdd(&(den_k[(xi1 - xb) * yspan * zspan + (yi0 - yb) * zspan + (zi1 - zb)]), w1x * w0y * w1z);
		atomicAdd(&(den_k[(xi1 - xb) * yspan * zspan + (yi1 - yb) * zspan + (zi0 - zb)]), w1x * w1y * w0z);
		atomicAdd(&(den_k[(xi1 - xb) * yspan * zspan + (yi1 - yb) * zspan + (zi1 - zb)]), w1x * w1y * w1z);
	}
}

void matter_power_spectrum_init() {
	vector<float> M;
	int xb, xe, yb, ye, zb, ze;
	int N = global().opts.parts_dim;
	particle_server pserv;
	const auto& parts = pserv.get_particle_set();
	auto tmp = parts.find_range(0, parts.size(), 0);
	xb = tmp.first.to_double() * N - 1;
	xe = tmp.second.to_double() * N + 2;
	tmp = parts.find_range(0, parts.size(), 1);
	yb = tmp.first.to_double() * N - 1;
	ye = tmp.second.to_double() * N + 2;
	tmp = parts.find_range(0, parts.size(), 2);
	zb = tmp.first.to_double() * N - 1;
	ze = tmp.second.to_double() * N + 2;
	int xspan = xe - xb;
	int yspan = ye - yb;
	int zspan = ze - zb;
	M.resize(xspan * yspan * zspan);
	double den0 = 1.0 / (N * N * N);
	for (int i = 0; i < xspan * yspan * zspan; i++) {
		M[i] = -den0;
	}
	cuda_set_device();
	cudaFuncAttributes attribs;
	CUDA_CHECK(cudaFuncGetAttributes(&attribs, matter_power_spectrum_kernel));
	int num_threads = attribs.maxThreadsPerBlock;
	int num_blocks;
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, matter_power_spectrum_kernel, num_threads, 0));
	num_blocks *= global().cuda.devices[0].multiProcessorCount;
	execute_kernel(matter_power_spectrum_kernel, parts, M.data(), xb, xe, yb, ye, zb, ze, N);
	std::vector<float> M2(M.size());
	M = decltype(M)();
	fourier3d_accumulate_real(xb, xe, yb, ye, zb, ze, std::move(M2));
}

/*********************************************************************************************************************************/

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
