#define SPHERICAL_HARMONIC_CPP

#include <cosmictiger/spherical_harmonic.hpp>

__constant__ sphericalYconstants gpu_spherical_constants;

void spherical_harmonics_init_gpu(const sphericalYconstants& constants) {
	CUDA_CHECK(cudaMemcpyToSymbol(gpu_spherical_constants, &constants, sizeof(sphericalYconstants)));
}
