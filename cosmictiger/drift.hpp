#pragma once

#include <cosmictiger/particle.hpp>

int drift_particles(particle_set parts, double dt, double a0, double*, double*, double*, double*, double, double);

void drift(particle_set *parts, double a1, double a2, double dtau);
