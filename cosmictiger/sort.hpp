#pragma once

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/particle.hpp>

enum sort_type {
	GPU_SORT, CPU_SORT
};

CUDA_KERNEL count_kernel(particle_set parts, size_t begin, size_t end, double xpos, int xdim, unsigned long long* count);
size_t count_particles(particle_set parts, size_t begin, size_t end, double xpos, int xdim);
std::function<bool(size_t*)> sort_particles(particle_set parts, size_t begin, size_t end, double xmid, int xdim);
CUDA_KERNEL gpu_sort_kernel(particle_set parts, size_t begin, unsigned long long* mid, size_t end, double xmid, int xdim,
		unsigned long long* bottom, unsigned long long*top);
size_t cpu_sort_kernel(particle_set parts, size_t begin, size_t end, double xmid, int xdim);
void stop_sort_daemon();

#ifndef __CUDACC__
#include <cosmictiger/hpx.hpp>
hpx::future<size_t> send_sort(particle_set parts, size_t begin, size_t end, double xmid, int xdim);
#endif
