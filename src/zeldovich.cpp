#include <cosmictiger/zeldovich.hpp>
#include <cosmictiger/particle_server.hpp>
#include <cosmictiger/hpx.hpp>

HPX_PLAIN_ACTION(denpow_to_phi1);

static float constrain_range(float r) {
	while (r >= 1.0f) {
		r -= 1.f;
	}
	while (r < 0.f) {
		r += 1.f;
	}
	return r;
}

float phi1_to_particles(int N, float box_size, float D1, float prefactor) {
	float dmax = 0.0f;
	particle_server pserv;
	particle_set& parts = pserv.get_particle_set();
	int xb = hpx_rank() * N / hpx_size();
	int xe = (hpx_rank() + 1) * N / hpx_size();
	int xspan = xe - xb;
	part_int count = xspan * N * N;
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
	auto phi1 = fourier3d_read_real(xb - 1, xe + 1, -1, N + 1, -1, N + 1);
	const float factor = -0.5f * D1 * N / box_size;
	for (int xi = xb; xi < xe; xi++) {
		const float x = (float(xi) + 0.5f) / float(N);
		for (int yi = 0; yi < N; yi++) {
			const float y = (float(yi) + 0.5f) / float(N);
			for (int zi = 0; zi < N; zi++) {
				const float z = (float(zi) + 0.5f) / float(N);
				part_int i0 = N * N * (xi - xb) + N * yi + zi;
				part_int i1px = (N + 2) * (N + 2) * (xi - xb + 2) + (N + 2) * (yi + 1) + (zi + 1);
				part_int i1mx = (N + 2) * (N + 2) * (xi - xb + 0) + (N + 2) * (yi + 1) + (zi + 1);
				part_int i1py = (N + 2) * (N + 2) * (xi - xb + 1) + (N + 2) * (yi + 2) + (zi + 1);
				part_int i1my = (N + 2) * (N + 2) * (xi - xb + 1) + (N + 2) * (yi + 0) + (zi + 1);
				part_int i1pz = (N + 2) * (N + 2) * (xi - xb + 1) + (N + 2) * (yi + 1) + (zi + 2);
				part_int i1mz = (N + 2) * (N + 2) * (xi - xb + 1) + (N + 2) * (yi + 1) + (zi - 0);
				const float dx = (phi1[i1px] - phi1[i1mx]) * factor;
				const float dy = (phi1[i1py] - phi1[i1my]) * factor;
				const float dz = (phi1[i1pz] - phi1[i1mz]) * factor;
				parts.pos(0, i0) += dx;
				parts.pos(1, i0) += dy;
				parts.pos(2, i0) += dz;
				dmax = std::max(dmax, dx);
				dmax = std::max(dmax, dy);
				dmax = std::max(dmax, dz);
			}
		}
	}
	printf("%e\n", dmax * N);
	return dmax;
}

void denpow_to_phi1(const interp_functor<float> den_k, int N, float box_size) {
	fourier3d_initialize(N);
	std::vector<hpx::future<void>> futs;
	if (hpx_rank() == 0) {
		for (int i = 1; i < hpx_size(); i++) {
			futs.push_back(hpx::async<denpow_to_phi1_action>(hpx_localities()[i], den_k, N, box_size));
		}
	}
	vector<cmplx> Y;
	int xbegin = hpx_rank() * N / hpx_size();
	int xend = (hpx_rank() + 1) * N / hpx_size();
	int xspan = xend - xbegin;
	Y.resize(xspan * N * N);
	const float factor = std::pow(box_size, -1.5);
	generate_random_normals(Y.data(), xspan * N * N, 42 + hpx_rank() * 1234);
	for (int i = xbegin; i < xend; i++) {
		int i0 = i < N / 2 ? i : i - N;
		float kx = 2.f * (float) M_PI / box_size * float(i0);
		for (int j = 0; j < N; j++) {
			int j0 = j < N / 2 ? j : j - N;
			float ky = 2.f * (float) M_PI / box_size * float(j0);
			for (int l = 0; l < N / 2; l++) {
				int l0 = l < N / 2 ? l : l - N;
				int i2 = i0 * i0 + j0 * j0 + l0 * l0;
				int index0 = N * (N * (i - xbegin) + j) + l;
				if (i2 > 0 && i2 < N * N / 4) {
					float kz = 2.f * (float) M_PI / box_size * float(l0);
					float k2 = kx * kx + ky * ky + kz * kz;
					float k = sqrt(k2);
					const float number = -std::sqrt(den_k(k)) * factor / k2;
					Y[index0].real() *= number;
					Y[index0].imag() *= number;
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
