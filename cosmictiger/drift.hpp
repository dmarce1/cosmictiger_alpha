#pragma once

#include <cosmictiger/particle.hpp>

int drift_particles(particle_sets parts, double dt, double a0, double*, double*, double*, double*, double, double);

void drift(particle_sets *parts, double a1, double a2, double dtau);
