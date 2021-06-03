#define SPHERICAL_HARMONIC_CPP
#include <cosmictiger/spherical_harmonic.hpp>

sphericalYconstants cpu_spherical_constants;

void spherical_harmonics_init() {
	float fact = 1.0;
	for (int i = 0; i < PMAX; i++) {
		cpu_spherical_constants.lp1inv[i] = 1.0f / (1.0f + i);
	}
	for (int i = 0; i < FACT_MAX; i++) {
		cpu_spherical_constants.factorial[i] = fact;
		fact *= float(i + 1);
	}
	for (int l = 0; l < PMAX; l++) {
		for (int m = 0; m <= l; m++) {
			const int index = ((l * (l + 1)) >> 1) + m;
			cpu_spherical_constants.bigA[index] = pow(-1.f, l) / sqrt(factorial(l - m) * factorial(l + m));
			cpu_spherical_constants.bigAinv[index] = 1.0f / cpu_spherical_constants.bigA[index];
			cpu_spherical_constants.Ynorm[index] = sqrt(factorial(l - (abs(m))) / factorial(l + abs(m)));
		}
	}
	cpu_spherical_constants.ipow[0] = cmplx(1, 0);
	for (int i = 1; i < 4; i++) {
		cpu_spherical_constants.ipow[i] = cmplx(0, 1) * cpu_spherical_constants.ipow[i - 1];
	}
	spherical_harmonics_init_gpu(cpu_spherical_constants);
}
