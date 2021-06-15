#pragma once

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/ewald_indices.hpp>
#include <cosmictiger/expansion.hpp>


inline bool anytrue(simd_float b) {
	return b.sum() != 0.0;
}

CUDA_EXPORT
inline bool anytrue(float b) {
	return b != 0.0;
}

template<class T>
CUDA_EXPORT int green_ewald(expansion<T> &D, array<T, NDIM> X) {
	return ewald_greens_function(D, X);
}
