#include <cosmictiger/map.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/simd.hpp>

#include <cosmictiger/math.hpp>
#include <atomic>
#include <unordered_map>

#include <chealpix.h>

template<typename T>
struct atomwrapper {
	std::atomic<T> _a;

	atomwrapper() :
			_a() {
	}

	atomwrapper(const std::atomic<T> &a) :
			_a(a.load()) {
	}

	atomwrapper(const atomwrapper &other) :
			_a(other._a.load()) {
	}

	atomwrapper& operator +=(T other) {
		_a += other;
		return *this;
	}

	atomwrapper &operator=(const atomwrapper &other) {
		_a.store(other._a.load());
	}

	operator T() const {
		return T(_a);
	}
};

using map_type = std::vector<atomwrapper<std::int64_t>>;

std::unordered_map<int, map_type> maps;

void prepare_map(int i) {
//	printf("preparing map %i\n", i);
	auto& map = maps[i];
	const auto npts = 12 * sqr(global().opts.map_size);
	map = map_type(npts, atomwrapper<int64_t>(0));
}

void save_map(int i) {
	const auto npts = 12 * sqr(global().opts.map_size);
	std::string filename = std::string("map.") + std::to_string(i) + ".hpx";
	std::vector<float> res;
	for (int j = 0; j < npts; j++) {
		res.push_back((float) int64_t(maps[i][j]));
	}
	FILE* fp = fopen(filename.c_str(), "wb");
	fwrite(res.data(), sizeof(float), npts, fp);
	fclose(fp);
}

void load_and_save_maps(double tau, double tau_max) {
	static int prepared_index = 0;
	static int saved_index = 0;
	const auto freq = global().opts.map_freq * tau_max;
	int imin = tau / freq + 1;
	int imax = (tau + 1.0) / freq + 1;
	for (int i = imin; i <= imax; i++) {
		auto iter = maps.find(i);
		if (iter == maps.end()) {
			prepare_map(i);
		}
	}
	for (auto i = maps.begin(); i != maps.end(); i++) {
		if (i->first < imin) {
			printf( "Saving map %i\n", i->first);
			save_map(i->first);
		}
	}
	auto i = maps.begin();
	while (i != maps.end()) {
		if (i->first < imin) {
			i = maps.erase(i);
		} else {
			i++;
		}
	}

}

simd_float images[NDIM] = { simd_float(0, -1, 0, -1, 0, -1, 0, -1), simd_float(0, 0, -1, -1, 0, 0, -1, -1), simd_float(
		0, 0, 0, 0, -1, -1, -1, -1) };

int map_add_part(const array<double, NDIM>& Y0, const array<double, NDIM>& Y1, double tau, double dtau,
		double tau_max) {
	static const auto map_freq = global().opts.map_freq * tau_max;
	static const auto map_freq_inv = 1.0 / map_freq;
	static const simd_float simd_c0 = simd_float(map_freq_inv);
	array<simd_float, NDIM> x0;
	array<simd_float, NDIM> x1;
	const simd_float simd_tau0 = simd_float(tau);
	const simd_float simd_tau1 = simd_float(tau + dtau);
	simd_float dist0;
	simd_float dist1;
	int rc = 0;
	double x20, x21, R20, R21;
	//static mutex_type mtx;
	for (int dim = 0; dim < NDIM; dim++) {
		x0[dim] = simd_float(Y0[dim]) + images[dim];
		x1[dim] = simd_float(Y1[dim]) + images[dim];
	}
	dist0 = sqrt(fma(x0[0], x0[0], fma(x0[1], x0[1], sqr(x0[2]))));
	dist1 = sqrt(fma(x1[0], x1[0], fma(x1[1], x1[1], sqr(x1[2]))));
	simd_float tau0 = simd_tau0 + dist0;
	simd_float tau1 = simd_tau1 + dist1;
	simd_int I0 = tau0 * simd_c0;
	simd_int I1 = tau1 * simd_c0;
	for (int ci = 0; ci < 8; ci++) {
		if (dist1[ci] <= 1.0) {
			const int i0 = I0[ci];
			const int i1 = I1[ci];
			if (i0 != i1) {
				for (int j = i0; j < i1; j++) {
					rc++;
					static const long Nside = global().opts.map_size;
					double r = dist1[ci];
					long ipring;
					array<double, NDIM> this_x;
					this_x[0] = x1[0][ci];
					this_x[1] = x1[1][ci];
					this_x[2] = x1[2][ci];
					vec2pix_ring(Nside, &this_x[0], &ipring);
					const std::int64_t dN = 100.0 / (r * r) + 0.5;
					maps[j + 1][ipring] += dN;
				}
			}
		}
	}
	return rc;
}
