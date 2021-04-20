#include <cosmictiger/map.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/simd.hpp>

#include <cosmictiger/math.hpp>
#include <atomic>
#include <unordered_map>

#include <chealpix.h>

static mutex_type mtx;
static std::unordered_map<int, map_type> maps;
//static std::stack<map_workspace> workspaces;

void maps_to_file(FILE*fp) {
	const auto npts = 12 * sqr(global().opts.map_size);
	int nmaps = maps.size();
	fwrite(&nmaps, sizeof(int), 1, fp);
	for (auto i = maps.begin(); i != maps.end(); i++) {
		fwrite(&(i->first), sizeof(int), 1, fp);
		fwrite(*(i->second), sizeof(float), npts, fp);
	}
}

void maps_from_file(FILE*fp) {
	const auto npts = 12 * sqr(global().opts.map_size);
	int nmaps;
	if (!feof(fp)) {
		FREAD(&nmaps, sizeof(int), 1, fp);
		for (int i = 0; i < nmaps; i++) {
			float* ptr;
			CUDA_MALLOC(ptr, npts);
			int index = 0;
			FREAD(&index, sizeof(int), 1, fp);
			maps[index] = std::make_shared<float*>(ptr);
			FREAD(*(maps[index]), sizeof(float), npts, fp);
		}
	}
}

map_workspace get_map_workspace() {
//	std::lock_guard<mutex_type> lock(mtx);
	return std::make_shared<std::unordered_map<int, std::array<vector<float>, NDIM>>>();
	/*	if (workspaces.size() == 0) {
	 auto ws = std::make_shared<std::unordered_map<int, std::array<vector<float>, NDIM>>>();
	 workspaces.push(ws);
	 }
	 auto ws = workspaces.top();
	 workspaces.pop();
	 return ws;*/
}


void prepare_map(int i) {
	printf("preparing map %i\n", i);
	auto& map = maps[i];
	const auto npts = 12 * sqr(global().opts.map_size);
	float* ptr;
	CUDA_MALLOC(ptr, npts);
	map = std::make_shared<float*>(ptr);
}

void cleanup_map_workspace(map_workspace ws) {
	static const long Nside = global().opts.map_size;
	for (auto i = ws->begin(); i != ws->end(); i++) {
		if (i->second[0].size()) {
			{
				std::lock_guard<mutex_type> lock(mtx);
				if (maps.find(i->first) == maps.end()) {
					prepare_map(i->first);
				}
			}
			healpix2_map(i->second[0], i->second[1], i->second[2], maps[i->first], Nside);
			for (int dim = 0; dim < NDIM; dim++) {
				i->second[dim] = vector<float>();
			}
		}
	}
//	std::lock_guard<mutex_type> lock(mtx);
//	workspaces.push(ws);
}

void save_map(int i) {
	const auto npts = 12 * sqr(global().opts.map_size);
	std::string filename = std::string("map.") + std::to_string(i) + ".hpx";
	FILE* fp = fopen(filename.c_str(), "wb");
	fwrite(*maps[i], sizeof(float), npts, fp);
	fclose(fp);
	CUDA_FREE(*maps[i]);
}

void load_and_save_maps(double tau, double tau_max) {
	static int prepared_index = 0;
	static int saved_index = 0;
	const auto freq = global().opts.map_freq * tau_max;
	int imin = tau / freq + 1;
	int imax = (tau + 1.2) / freq + 1;
//	printf( "imin %i max %i\n", imin, imax);
	for (auto i = maps.begin(); i != maps.end(); i++) {
		if (i->first < imin) {
			timer tm;
			tm.start();
			printf("                                               \rSaving map %i\n", i->first);
			save_map(i->first);
			tm.stop();
			printf( "Done. Took %e s\n", tm.read());
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

int map_add_part(const array<double, NDIM>& Y0, const array<double, NDIM>& Y1, double tau, double dtau, double tau_max,
		map_workspace& ws) {
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
					auto& this_ws = (*ws)[j + 1];
					for (int dim = 0; dim < NDIM; dim++) {
						this_ws[dim].push_back(x1[dim][ci]);
					}
					/*				vec2pix_ring(Nside, &this_x[0], &ipring);
					 const std::int64_t dN = 100.0 / (r * r) + 0.5;
					 maps[j + 1][ipring] += dN;*/
				}
			}
		}
	}
	return rc;
}
