#pragma once


#include <cosmictiger/cuda.hpp>
#include <cosmictiger/particle.hpp>

CUDA_KERNEL count_kernel(particle_set parts, size_t begin, size_t end, fixed32 xpos, int xdim, size_t* count );
size_t count_particles(particle_set parts, size_t begin, size_t end, fixed32 xpos, int xdim);
fixed32 sort_particles(particle_set parts, size_t begin, size_t end, fixed32 xmin, fixed32 xmax, int xdim);
CUDA_KERNEL gpu_sort_kernel(particle_set parts, size_t begin, size_t end, fixed32 xmid, int xdim,
		unsigned long long* bottom);
