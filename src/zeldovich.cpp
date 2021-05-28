#include <cosmictiger/zeldovich.hpp>
#include <cosmictiger/hpx.hpp>


HPX_PLAIN_ACTION(denpow_to_phi1);

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
		for (int j = 0; j < N; j++) {
			int i0 = i < N / 2 ? i : i - N;
			int j0 = j < N / 2 ? j : j - N;
			float kx = 2.f * (float) M_PI / box_size * float(i0);
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
				}
			}
		}
	}
	fourier3d_accumulate(xbegin, xend, 0, N, 0, N, std::move(Y));
	hpx::wait_all(futs.begin(), futs.end());
	fourier3d_mirror();
	fourier3d_execute();
}
