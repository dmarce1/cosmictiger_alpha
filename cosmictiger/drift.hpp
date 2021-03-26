#pragma once

#include <cosmictiger/particle.hpp>

int drift_particles(particle_set parts, double dt, double a0, double a1, double*, double*, double*, double*, double, double);
CUDA_KERNEL drift_kernel(particle_set parts, double dt, double a0, double a1);
