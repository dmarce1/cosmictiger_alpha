#include <cosmictiger/zeldovich.hpp>
#include <cosmictiger/particle_server.hpp>
#include <cosmictiger/hpx.hpp>

HPX_PLAIN_ACTION(_2lpt);
HPX_PLAIN_ACTION(phi1_to_particles);

static double constrain_range(double r) {
	while (r >= 1.0) {
		r -= 1.;
	}
	while (r < 0.) {
		r += 1.;
	}
	return r;
}

float phi1_to_particles(int N, float box_size, float D1, float prefactor, int dim) {
	std::vector<hpx::future<void>> futs;
	if (hpx_rank() == 0) {
		for (int i = 1; i < hpx_size(); i++) {
			futs.push_back(hpx::async<phi1_to_particles_action>(hpx_localities()[i], N, box_size, D1, prefactor, dim));
		}
	}
	double dmax = 0.0f;
	particle_server pserv;
	particle_set& parts = pserv.get_particle_set();
	int xb = hpx_rank() * N / hpx_size();
	int xe = (hpx_rank() + 1) * N / hpx_size();
	int xspan = xe - xb;
	part_int count = xspan * N * N;
	if (dim == 0) {
		parts.resize(count);
		for (int xi = xb; xi < xe; xi++) {
			const float x = (float(xi) + 0.5f) / float(N);
			for (int yi = 0; yi < N; yi++) {
				const float y = (float(yi) + 0.5f) / float(N);
				for (int zi = 0; zi < N; zi++) {
					const float z = (float(zi) + 0.5f) / float(N);
					part_int index = N * N * (xi - xb) + N * yi + zi;
					parts.pos(0, index) = x;
					parts.pos(1, index) = y;
					parts.pos(2, index) = z;
				}
			}
		}
	}
	auto phi1 = fourier3d_read_real(xb, xe, 0, N, 0, N);
	const float factor = -D1 / box_size;
	for (int xi = xb; xi < xe; xi++) {
		const float x = (float(xi) + 0.5f) / float(N);
		for (int yi = 0; yi < N; yi++) {
			const float y = (float(yi) + 0.5f) / float(N);
			for (int zi = 0; zi < N; zi++) {
				const float z = (float(zi) + 0.5f) / float(N);
				part_int i0 = N * N * (xi - xb) + N * yi + zi;
				const double dx = phi1[i0] * factor;
				if (std::abs(dx) > 1.0) {
					printf("phi1 is huge %e\n", dx);
				}
				const double x = parts.pos(dim, i0).to_double() + dx;
				constrain_range(x);
				parts.pos(dim, i0) = x;
				dmax = std::max(dmax, dx);
			}
		}
	}
	printf("%e\n", dmax * N);
	hpx::wait_all(futs.begin(), futs.end());
	return dmax;
}

void _2lpt(const interp_functor<float> den_k, int N, float box_size, int dim1, int dim2, int seed) {
	fourier3d_initialize(N);
	std::vector<hpx::future<void>> futs;
	if (hpx_rank() == 0) {
		for (int i = 1; i < hpx_size(); i++) {
			futs.push_back(hpx::async<_2lpt_action>(hpx_localities()[i], den_k, N, box_size, dim1, dim2, seed));
		}
	}
	vector<cmplx> Y;
	int xbegin = hpx_rank() * N / hpx_size();
	int xend = (hpx_rank() + 1) * N / hpx_size();
	int xspan = xend - xbegin;
	Y.resize(xspan * N * N);
	const float factor = std::pow(box_size, -1.5);
	generate_random_normals(Y.data(), xspan * N * N, seed + hpx_rank() * 4321);
	for (int i = xbegin; i < xend; i++) {
		int i0 = i < N / 2 ? i : i - N;
		float kx = 2.f * (float) M_PI / box_size * float(i0);
		for (int j = 0; j < N; j++) {
			int j0 = j < N / 2 ? j : j - N;
			float ky = 2.f * (float) M_PI / box_size * float(j0);
			for (int l = 0; l < N; l++) {
				int l0 = l < N / 2 ? l : l - N;
				int i2 = i0 * i0 + j0 * j0 + l0 * l0;
				int index0 = N * (N * (i - xbegin) + j) + l;
				if (i2 > 0 && i2 < N * N / 4) {
					float kz = 2.f * (float) M_PI / box_size * float(l0);
					float k2 = kx * kx + ky * ky + kz * kz;
					float k = std::sqrt(kx * kx + ky * ky + kz * kz);
					const cmplx K[NDIM + 1] = { {kx,0}, {ky,0}, {kz,0}, {0,-1} };
					const cmplx number = std::sqrt(den_k(k)) * factor * K[dim1] * K[dim2] / k2;
					Y[index0] = Y[index0] * number;
				} else {
					Y[index0].real() = Y[index0].imag() = 0.0;
				}
			}
		}
	}
	fourier3d_accumulate(xbegin, xend, 0, N, 0, N, std::move(Y));
	hpx::wait_all(futs.begin(), futs.end());
	fourier3d_mirror();
	fourier3d_execute();
}
