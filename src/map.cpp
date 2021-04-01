#include <cosmictiger/map.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/global.hpp>

#include <cosmictiger/math.hpp>
#include <atomic>
#include <unordered_map>

#include <silo.h>
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

int map_add_part(const array<double, NDIM>& Y0, const array<double, NDIM>& Y1, double tau, double dtau,
		double tau_max) {
	static const auto map_freq = global().opts.map_freq * tau_max;
	static const auto map_freq_inv = 1.0 / map_freq;
	array<double, NDIM> x0;
	array<double, NDIM> x1;
	int rc = 0;
	double x20, x21, R20, R21;
	//static mutex_type mtx;
	for (int xi = -1; xi <= 0; xi++) {
		x0[0] = Y0[0] + double(xi);
		x1[0] = Y1[0] + double(xi);
		x20 = sqr(x0[0]);
		x21 = sqr(x1[0]);
		for (int yi = -1; yi <= 0; yi++) {
			x0[1] = Y0[1] + double(yi);
			x1[1] = Y1[1] + double(yi);
			R20 = sqr(x0[1]) + x20;
			R21 = sqr(x1[1]) + x21;
			for (int zi = -1; zi <= 0; zi++) {
				x0[2] = Y0[2] + double(zi);
				x1[2] = Y1[2] + double(zi);
				const auto r1 = std::sqrt(R21 + sqr(x1[2]));
				if (r1 <= 1.0) {
					const auto r0 = std::sqrt(R20 + sqr(x0[2]));
					const auto tau0 = tau + r0;
					const auto tau1 = tau + dtau + r1;

					const int i0 = tau0 * map_freq_inv;
					const int i1 = tau1 * map_freq_inv;
					if (i0 != i1) {
						for (int j = i0; j < i1; j++) {
							rc++;
							static const long Nside = global().opts.map_size;
							for (int dim = 0; dim < NDIM; dim++) {
								x0[dim] = 0.5 * (x0[dim] + x1[dim]);
							}
							double r = fma(x0[0], x0[0], fma(x0[1], x0[1], sqr(x0[2])));
							long ipring;
							vec2pix_ring(Nside, &x0[0], &ipring);
							const std::int64_t dN = 100.0 / (r * r) + 0.5;
							maps[j + 1][ipring] += dN;
						}
					}
				}
			}
		}
	}
	return rc;
}
