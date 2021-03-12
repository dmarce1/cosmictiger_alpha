#pragma once

#include <cosmictiger/particle.hpp>

void drift_particles(particle_set parts, double dt, double a0, double a1);
CUDA_KERNEL drift_kernel(particle_set parts, double dt, double a0, double a1);
