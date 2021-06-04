#define SPHERICAL_HARMONIC_CPP

#include <cosmictiger/spherical_harmonic.hpp>

__managed__ sphericalYconstants gpu_spherical_constants;

void spherical_harmonics_init_gpu(const sphericalYconstants& constants) {
	cuda_set_device();

	gpu_spherical_constants=constants;
}
