#include <cosmictiger/power.hpp>
#include <cosmictiger/constants.hpp>

CUDA_EXPORT
float eisenstein_and_hu(float k, float omega_c, float omega_b, float hubble) {

	/*** PARTS OF THIS WERE ADAPTED FROM N-GENIC ***/

	float q, theta, ommh2, a, s, gamma, L0, C0;
	float tmp;
	float omegam, ombh2;

	omegam = omega_b + omega_c;
	ombh2 = omega_b * hubble * hubble;

	k *= (3.085678e24f / constants::mpc_to_cm); /* convert to h/Mpc */

	theta = 2.728f / 2.7f;
	ommh2 = omegam * hubble * hubble;
	s = 44.5f * logf(9.83f / ommh2) / sqrtf(1.f + 10.f * expf(0.75f * logf(ombh2))) * hubble;
	a = 1.f - 0.328f * logf(431.f * ommh2) * ombh2 / ommh2 + 0.380f * logf(22.3f * ommh2) * (ombh2 / ommh2) * (ombh2 / ommh2);
	gamma = a + (1.f - a) / (1.f + exp(4.f * logf(0.43f * k * s)));
	gamma *= omegam * hubble;
	q = k * theta * theta / gamma;
	L0 = logf(2.f * 2.7182818284590f + 1.8f * q);
	C0 = 14.2f + 731.f / (1.f + 62.5f * q);
	tmp = L0 / (L0 + C0 * q * q);
	return k * tmp * tmp;
}
