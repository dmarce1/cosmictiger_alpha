#include <cosmictiger/map.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/global.hpp>

#include <cosmictiger/math.hpp>
#include <atomic>
#include <unordered_map>

#include <silo.h>

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
	const auto xdim = global().opts.map_size * 2;
	const auto ydim = xdim / 2;
	map = map_type(xdim * ydim, atomwrapper<int64_t>(0));
}

void save_map(int i) {
	const auto xdim = global().opts.map_size * 2;
	const auto ydim = xdim / 2;
	std::string filename = std::string("map.") + std::to_string(i) + ".silo";
	auto db = DBCreate(filename.c_str(), DB_CLOBBER, DB_LOCAL, NULL, DB_PDB);
	const map_type& map = maps[i];
	std::vector<double> vals;
	std::vector<double> x, y;
	for (int j = 0; j < map.size(); j++) {
		vals.push_back((double) int64_t(map[j]));
	}
	for (int j = 0; j <= xdim; j++) {
		x.push_back(2.0 * j / xdim - 1.0);
	}
	for (int j = 0; j <= ydim; j++) {
		y.push_back(double(j) / ydim - 0.5);
	}

	constexpr int ndim = 2;
	const int dims1[ndim] = { xdim + 1, ydim + 1 };
	const int dims2[ndim] = { xdim, ydim };
	const double* coords[ndim] = { x.data(), y.data() };
	const char* coord_names[ndim] = { "x", "y" };
	DBPutQuadmesh(db, "mesh", coord_names, coords, dims1, ndim, DB_DOUBLE, DB_COLLINEAR, nullptr);
	DBPutQuadvar1(db, "intensity", "mesh", vals.data(), dims2, ndim, nullptr, 0, DB_DOUBLE, DB_ZONECENT, nullptr);
	DBClose(db);
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

void map_coords(double& x, double& y, double lambda, double phi) {
	double theta = phi;
	double theta0;
	static const double res = 1.0 / global().opts.map_size;
	int iters = 0;
	do {
		theta0 = theta;
		theta -= (2.0 * theta + std::sin(2.0 * theta) - M_PI * std::sin(phi)) / (2.0 + 2.0 * std::cos(2.0 * theta));
		iters++;
	} while (std::abs(theta0 - theta) > res / 100.0);
	x = lambda * std::cos(theta) / (M_PI);
	y = std::sin(theta);
//	printf( "%e %e\n", x, y);
//	printf( "%i\n", iters);
}

int map_add_part(array<double, NDIM> Y0, array<double, NDIM> Y1, double tau, double dtau, double tau_max) {
	array<double, NDIM> x0;
	array<double, NDIM> x1;
	double tau0, tau1;
	int rc = 0;
	static int nmapped = 0;
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
					const auto map_freq = global().opts.map_freq * tau_max;
					tau0 = tau + r0;
					tau1 = tau + dtau + r1;
					const int i0 = tau0 / map_freq;
					const int i1 = tau1 / map_freq;
					for (int j = i0; j < i1; j++) {
						rc++;
						const auto xdim = global().opts.map_size * 2;
						const auto ydim = xdim / 2;
						auto x = (x0[0] + x1[0]) * 0.5;
						auto y = (x0[1] + x1[1]) * 0.5;
						const auto z = (x0[2] + x1[2]) * 0.5;
						const auto R = std::sqrt(x * x + y * y);
						const auto r = std::sqrt(R * R + z * z);
						if (r > 1.0e-3) {
							const auto zor = z / r;
							const auto theta = std::asin(zor);
							const auto phi = std::atan2(y, x);
							map_coords(x, y, phi, theta);
							int ix = (x + 1.0) / 2.0 * xdim + 0.5;
							int iy = (y + 1.0) / 2.0 * ydim + 0.5;
							ix = std::min(ix, xdim - 1);
							iy = std::min(iy, ydim - 1);
							ix = std::max(ix, 0);
							iy = std::max(iy, 0);
							const std::int64_t dN = 100.0 / (r * r) + 0.5;
							/*{
								std::lock_guard<mutex_type> lock(mtx);
								FILE* fp = fopen("test.dat", "at");
								fprintf(fp, "%i %i\n", ix, iy);
								fclose(fp);
							}*/
							if (maps.find(j + 1) == maps.end()) {
								printf("map %i not found\n", j + 1);
								abort();
							}
							maps[j + 1][ix + xdim * iy] += dN;
						}
					}
				}
			}
		}
	}
	return rc;
}
