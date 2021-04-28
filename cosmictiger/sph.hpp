#pragma once

#include <cosmictiger/hydro_particle.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/tree_database.hpp>

enum sph_neighbors_type {
	ACTIVE, SEMIACTIVE
};

struct sph_neighbor_params_type {
	vector<tree_ptr> next_checks;
	vector<tree_ptr> opened_checks;
	stack_vector<tree_ptr> checks;
	hydro_particle_set parts;
	tree_ptr self;
	int depth;
	size_t block_cutoff;
	int rung;
	sph_neighbors_type type;
	float search_cushion;
};

hpx::future<bool> sph_neighbors(sph_neighbor_params_type*);

CUDA_EXPORT
inline float W(float q) {
	const float onemq = 1.f - q;
	if (q < 0.5f) {
		return 2.546479089f * fmaf(-6.f, sqr(q) * onemq, 1.f);
	} else if (q < 1.f) {
		return 5.092958179f * sqr(onemq) * onemq;
	} else {
		return 0.f;
	}
}

CUDA_EXPORT
inline float dWdq(float q) {
	if (q < 0.5f) {
		return -15.278874537f * q * fmaf(-3.f, q, 2.f);
	} else if (q < 1.f) {
		return -15.278874537f * sqr(1.f - q);
	} else {
		return 0.f;
	}
}

CUDA_EXPORT
inline float W(float r, float hinv) {
	return W(r * hinv) * hinv * sqr(hinv);
}

CUDA_EXPORT
inline float dWdr(float r, float hinv) {
	return dWdr(r, hinv) * sqr(sqr(hinv));
}

CUDA_EXPORT
inline float dWdh(float r, float hinv) {
	return -dWdr(r, hinv) * r * sqr(sqr(hinv)) * hinv;
}

