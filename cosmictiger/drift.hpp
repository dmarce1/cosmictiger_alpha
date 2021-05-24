#pragma once

#include <cosmictiger/particle.hpp>



struct drift_return {
	double ekin;
	double momx;
	double momy;
	double momz;
	size_t map_cnt;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & ekin;
		arc & momx;
		arc & momy;
		arc & momz;
		arc & map_cnt;
	}
};

drift_return drift(double dt, double a, double tau, double tau_max);

int drift_particles(particle_set parts, double dt, double a0, double*, double*, double*, double*, double, double);
