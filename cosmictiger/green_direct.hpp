#pragma once

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/expansion.hpp>

#define DSCALE 1e4
#define DSCALE2 1e8
#define DSCALE3 1e12
#define DSCALE4 1e16
#define DSCALE5 1e20
#define RCUT 1e-4
#define RCUT2 1e-8

CUDA_EXPORT inline bool any_true(bool a) {
	return a;
}

inline bool any_true(simd_float a) {
	return a.sum();
}

template<class T>
CUDA_EXPORT int green_direct(expansion<T> &D, array<T, NDIM> dX, T rmin = 0.f) {
	D = direct_greens_function<T, LORDER>(dX);
	return 0;
}
