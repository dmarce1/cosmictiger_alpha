#include <cosmictiger/direct.hpp>
#include <cosmictiger/particle_server.hpp>
#include <cosmictiger/gravity.hpp>

CUDA_KERNEL cuda_pp_ewald_interactions(particle_set parts, fixed32*x, fixed32* y, fixed32* z, gforce *forces, float GM,
		float h) {
	const int &tid = threadIdx.x;
	const int &bid = blockIdx.x;
	const auto hinv = 1.f / h * 0.5;
	const auto h3inv = 1.f / h / h / h * 0.125;
	const auto h2 = 4.0 * h * h;
	array<fixed32, NDIM> sink;
	sink[0] = x[bid];
	sink[1] = y[bid];
	sink[2] = z[bid];
	__shared__ array<array<double, EWALD_BLOCK_SIZE>, NDIM> f;
	__shared__ array<double, EWALD_BLOCK_SIZE> phi;
	for (int dim = 0; dim < NDIM; dim++) {
		f[dim][tid] = 0.0;
	}
	phi[tid] = 0.0;
	for (part_int source = tid; source < parts.size(); source += EWALD_BLOCK_SIZE) {
		array<float, NDIM> X;
		for (int dim = 0; dim < NDIM; dim++) {
			const auto a = sink[dim];
			const auto b = parts.pos(dim, source);
			X[dim] = distance(a, b);
		}
		const auto r2 = fmaf(X[0], X[0], fmaf(X[1], X[1], sqr(X[2])));

#ifdef	PERIODIC_OFF
		constexpr bool periodic_off = true;
#else
		constexpr bool periodic_off = false;
#endif
		if (r2 < h2 || periodic_off) {
			float r3inv, r1inv;
			if (r2 >= h2) {
				r1inv = rsqrt(r2);
				r3inv = r1inv * r1inv * r1inv;
			} else {
				const float r1oh1 = sqrtf(r2) * hinv;              // 1 + FLOP_SQRT
				const float r2oh2 = r1oh1 * r1oh1;           // 1
				r3inv = float(-35.0 / 16.0);
				r3inv = fmaf(r3inv, r2oh2, float(135.0 / 16.0));
				r3inv = fmaf(r3inv, r2oh2, float(-189.0 / 16.0));
				r3inv = fmaf(r3inv, r2oh2, float(105.0 / 16.0));
				r3inv *= h3inv;
				r1inv = float(35.0 / 128.0);
				r1inv = fmaf(r1inv, r2oh2, float(-45.0 / 32.0));
				r1inv = fmaf(r1inv, r2oh2, float(189.0 / 64.0));
				r1inv = fmaf(r1inv, r2oh2, float(-105.0 / 32.0));
				r1inv = fmaf(r1inv, r2oh2, float(315.0 / 128.0));
				r1inv *= hinv;
			}
			phi[tid] -= r1inv;
			for (int dim = 0; dim < NDIM; dim++) {
				f[dim][tid] -= X[dim] * r3inv;
			}
		} else {
			const ewald_const econst;
			for (int i = 0; i < econst.nreal(); i++) {
				const auto n = econst.real_index(i);
				array<float, NDIM> dx;
				for (int dim = 0; dim < NDIM; dim++) {
					dx[dim] = X[dim] - n[dim];
				}
				const float r2 = sqr(dx[0]) + sqr(dx[1]) + sqr(dx[2]);
				if (r2 < (EWALD_REAL_CUTOFF2)) {  // 1
					const float r = sqrt(r2);  // 1
					const float rinv = 1.f / r;  // 2
					const float r2inv = rinv * rinv;  // 1
					const float r3inv = r2inv * rinv;  // 1
					const float exp0 = expf(-4.f * r2);  // 26
					const float erfc0 = erfcf(2.f * r);  // 10
					const float expfactor = 4.0 / sqrt(M_PI) * r * exp0;  // 2
					const float d0 = -erfc0 * rinv;
					const float d1 = (expfactor + erfc0) * r3inv;  // 2
					phi[tid] += d0;
					for (int dim = 0; dim < NDIM; dim++) {
						f[dim][tid] -= dx[dim] * d1;
					}
				}
			}
			for (int i = 0; i < econst.nfour(); i++) {
				const auto &h = econst.four_index(i);
				const auto &hpart = econst.four_expansion(i);
				const float hdotx = h[0] * X[0] + h[1] * X[1] + h[2] * X[2];
				float co = cosf(2.0 * M_PI * hdotx);
				float so = sinf(2.0 * M_PI * hdotx);
				phi[tid] += hpart(0, 0, 0) * co;
				f[0][tid] -= hpart(1, 0, 0) * so;
				f[1][tid] -= hpart(0, 1, 0) * so;
				f[2][tid] -= hpart(0, 0, 1) * so;
			}
			phi[tid] += float(M_PI / 4.f);
		}
	}
	__syncthreads();
	for (int P = EWALD_BLOCK_SIZE / 2; P >= 1; P /= 2) {
		if (tid < P) {
			for (int dim = 0; dim < NDIM; dim++) {
				f[dim][tid] += f[dim][tid + P];
			}
			phi[tid] += phi[tid + P];
		}
		__syncthreads();
	}
	if (tid == 0) {
		for (int dim = 0; dim < NDIM; dim++) {
			f[dim][0] *= GM;
			forces[bid].f[dim] = f[dim][0];
		}
		phi[0] *= GM;
		forces[bid].phi = phi[0];
	}
}

std::vector<gforce> cuda_direct(std::vector<std::array<fixed32, NDIM>> pts) {
	particle_server pserv;
	auto& parts = pserv.get_particle_set();
	std::vector<gforce> forces(pts.size());
	gforce* dforces;
	fixed32* x, *y, *z;
	CUDA_MALLOC(x, pts.size());
	CUDA_MALLOC(y, pts.size());
	CUDA_MALLOC(z, pts.size());
	CUDA_MALLOC(dforces, pts.size());
	for (int i = 0; i < pts.size(); i++) {
		x[i] = pts[i][0];
		y[i] = pts[i][1];
		z[i] = pts[i][2];
	}
	cuda_pp_ewald_interactions<<<pts.size(),EWALD_BLOCK_SIZE>>>(parts, x, y, z, dforces, global().opts.G * global().opts.M, global().opts.hsoft);
	CUDA_CHECK(cudaDeviceSynchronize());
	for (int i = 0; i < pts.size(); i++) {
		forces[i] = dforces[i];
	}
	CUDA_FREE(dforces);
	CUDA_FREE(x);
	CUDA_FREE(y);
	CUDA_FREE(z);
	return forces;

}

