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
	std::vector<hpx::future<float>> futs;
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
				parts.vel(dim, i0) = prefactor * dx;
				dmax = std::max(dmax, dx);
			}
		}
	}
	dmax *= N;
	for (auto& f : futs) {
		dmax = std::max(f.get(), (float) dmax);
	}
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
	printf("2lpt on %i\n", hpx_rank());
	vector<cmplx> Y;
	int xbegin = hpx_rank() * N / hpx_size();
	int xend = (hpx_rank() + 1) * N / hpx_size();
	int xspan = xend - xbegin;
	Y.resize(xspan * N * N);
	const float factor = std::pow(box_size, -1.5);
	generate_random_normals(Y.data(), xspan * N * N, seed + hpx_rank() * 4321);
	execute_2lpt_kernel(Y.data(), xbegin, xend, den_k, N, box_size, dim1, dim2);
	fourier3d_accumulate(xbegin, xend, 0, N, 0, N, std::move(Y));
	hpx::wait_all(futs.begin(), futs.end());
	if (hpx_rank() == 0) {
		fourier3d_mirror();
		fourier3d_execute(true);
	}
}
