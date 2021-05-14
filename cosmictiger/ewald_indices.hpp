#pragma once

#include <cosmictiger/array.hpp>
#include <cosmictiger/expansion.hpp>
#include <cosmictiger/defs.hpp>

struct ewald_const {
	CUDA_EXPORT static int nfour();
	CUDA_EXPORT static int nreal();
	static void init();
	static void init_gpu();
	CUDA_EXPORT static const array<float,NDIM>& real_index(int i);
	CUDA_EXPORT static const array<float,NDIM>& four_index(int i);
	CUDA_EXPORT static const expansion<float>& four_expansion(int i);
};
