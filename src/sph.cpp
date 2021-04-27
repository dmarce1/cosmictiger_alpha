#include <cosmictiger/sph.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/particle_sets.hpp>

struct gpu_sph_neighbors {

};

static std::vector<gpu_sph_neighbors> gpu_queue;
static mutex_type mutex;
static std::atomic<size_t> parts_covered;

struct sph_find_ranges_params {
	tree_ptr self;
	hydro_particle_set& parts;
	float factor;
};

void send_sph_neighbors_to_gpu() {

}
hpx::future<bool> tree_ptr::sph_neighbors(sph_neighbor_params_type*, bool trythread) {

}

hpx::future<bool> sph_neighbors(sph_neighbor_params_type* params_ptr) {
	sph_neighbor_params_type& params = *params_ptr;
	auto& parts = params.parts;
	tree_ptr self = params.self;
	if (params.depth == 0) {
		parts_covered = 0;
		gpu_queue.resize(0);
		size_t dummy, total_mem;
		CUDA_CHECK(cudaMemGetInfo(&dummy, &total_mem));
		total_mem /= 8;
		size_t used_mem = (sizeof(float) + sizeof(fixed32) * NDIM) * parts.size() + tree_data_bytes_used();
		double oversubscription = std::max(2.0, (double) used_mem / total_mem);
		const int block_count = oversubscription * global().cuda_kick_occupancy
				* global().cuda.devices[0].multiProcessorCount + 0.5;
		const auto active_count = self.get_parts(BARY_SET).second - self.get_parts(BARY_SET).first;
		params.block_cutoff = std::max(active_count / block_count, (size_t) 1);
	}

	auto& checks = params.checks;
	auto& next_checks = params.next_checks;
	auto& opened_checks = params.opened_checks;

	auto myrange = self.get_range();
	const auto iamleaf = self.is_leaf();
	opened_checks.resize(0);
	do {
		next_checks.resize(0);
		for (int i = 0; i < checks.size(); i++) {
			const auto other_range = checks[i].get_range();
			if (myrange.intersects(other_range)) {
				if (checks[i].is_leaf()) {
					opened_checks.push_back(checks[i]);
				} else {
					next_checks.push_back(checks[i]);
				}
			}
		}
		checks.resize(NCHILD * next_checks.size());
		for (int i = 0; i < next_checks.size(); i++) {
			const auto children = next_checks[i].get_children();
			checks[NCHILD * i + LEFT] = children[LEFT];
			checks[NCHILD * i + RIGHT] = children[RIGHT];
		}
	} while (iamleaf && checks.size());
	for (int i = 0; i < opened_checks.size(); i++) {
		checks.push(opened_checks[i]);
	}
	if (iamleaf) {
		const float c0 = (4.0 * M_PI) / 3.0;
		const auto myparts = self.get_parts(CDM_SET);
		float err = 0.0;
		for (int j = myparts.first; j != myparts.second; j++) {
			do {
				float h = parts.smooth_len(j);
				float N = 0.0;
				float dNdh = 0.0;
				const float hinv = 1.0f / h;
				const float h3 = h * sqr(h);
				for (int i = 0; i < checks.size(); i++) {
					const auto other_parts = checks[i].get_parts(CDM_SET);
					for (int k = other_parts.first; k != other_parts.second; k++) {
						float dx0, dx1, dx2;
						dx0 = distance(parts.pos(0, j), parts.pos(0, k));
						dx1 = distance(parts.pos(1, j), parts.pos(1, k));
						dx2 = distance(parts.pos(2, j), parts.pos(2, k));
						const float r2 = fma(dx0, dx0, fma(dx1, dx1, sqr(dx2)));
						const float r = sqrt(r2);
						N += c0 * h3 * W(r, hinv);
						dNdh += c0 * h3 * dWdh(r, hinv);
					}
				}
				dNdh += 3.0f * N * hinv;
				float hnp1 = h - (N - NNEIGHBORS) / dNdh;
				err = fabs(h - hnp1) * hinv;
			} while (err > SPH_TOLER);
		}
		parts_covered += myparts.second - myparts.first;
		if (parts_covered == params.parts.size()) {
			send_sph_neighbors_to_gpu();
		}
		return hpx::make_ready_future(false);
	} else {
		std::array<hpx::future<bool>, NCHILD> futs;
		auto mychildren = self.get_children();
		params.checks.push_top();
		params.self = mychildren[LEFT];
		params.depth++;
		futs[LEFT] = mychildren[LEFT].sph_neighbors(params_ptr, true);
		params.checks.pop_top();
		params.self = mychildren[RIGHT];
		futs[RIGHT] = mychildren[RIGHT].sph_neighbors(params_ptr, false);
		params.depth--;
		return hpx::when_all(futs.begin(), futs.end()).then([self](hpx::future<std::vector<hpx::future<bool>>> futfut) {
			auto futs = futfut.get();
			bool rc1 = futs[LEFT].get();
			bool rc2 = futs[RIGHT].get();
			return rc1 || rc2;
		});
	}

}
