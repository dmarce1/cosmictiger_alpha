#pragma once


#include <cosmictiger/cuda.hpp>
#include <cosmictiger/particle.hpp>

CUDA_KERNEL count_kernel(particle_set parts, size_t begin, size_t end, double xpos, int xdim, size_t* count );
size_t count_particles(particle_set parts, size_t begin, size_t end, double xpos, int xdim);
size_t sort_particles(particle_set parts, size_t begin, size_t end, double xmid, int xdim);
CUDA_KERNEL gpu_sort_kernel(particle_set parts, size_t begin, size_t mid, size_t end, double xmid, int xdim,
		unsigned long long* bottom, unsigned long long*top);
