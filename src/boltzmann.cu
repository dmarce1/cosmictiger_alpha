#include <cosmictiger/boltzmann.hpp>
#include <cosmictiger/memory.hpp>

__managed__ zero_order_universe* zero_verse;

void set_zeroverse(zero_order_universe* z) {
	CUDA_MALLOC(zero_verse, 1);
	new (zero_verse) zero_order_universe(*z);
}

__device__ float sigma8_integrand::operator()(float x) const {
	const float R = 8 / littleh;
	const float c0 = float(9) / (2. * float(M_PI) * float(M_PI)) / powf(R, 6);
	float k = expf(x);
	cos_state U;
	einstein_boltzmann_init(&U, uni, k, 1., uni->amin);
	einstein_boltzmann(&U, uni, k, uni->amin, 1.);
	float oc = zero_verse->params.omega_c;
	float ob = zero_verse->params.omega_b;
	float tmp = (oc * U[deltaci] + ob * U[deltabi]) / (oc + ob);
	float P = tmp * tmp;
	tmp = (SIN(k*R) - k * R * COS(k * R));
	return c0 * P * tmp * tmp * powf(k, -3);
}

__device__ void einstein_boltzmann_init(cos_state* uptr, const zero_order_universe* uni_ptr, float k,
		float normalization, float a) {
	cos_state& U = *uptr;
	const zero_order_universe& uni = *uni_ptr;
	float Oc, Ob, Ogam, Onu, Or;
	uni.compute_radiation_fractions(Ogam, Onu, a);
	uni.compute_matter_fractions(Oc, Ob, a);
	Or = Ogam + Onu;
	float hubble = hubble_function(a, uni.params.hubble, uni.params.omega_b + uni.params.omega_c,
			uni.params.omega_gam + uni.params.omega_nu);
	float eps = k / (a * hubble);
	float C = (float) 1.0 * powf(eps, (float) -1.5) * normalization;
	float Rnu = Onu / Or;
	U[taui] = (float) 1.0 / (a * hubble);
	U[deltanui] = U[deltagami] = -(float) 2.0 / (float) 3.0 * C * eps * eps;
	U[deltabi] = U[deltaci] = (float) 3.0 / (float) 4.0 * U[deltagami];
	U[thetabi] = U[thetagami] = -C / (float) 18.0 * eps * eps * eps;
	U[thetanui] = ((float) 23 + (float) 4 * Rnu) / ((float) 15 + (float) 4 * Rnu) * U[thetagami];
	U[N2i] = (float) 0.5 * ((float) 4.0 * C) / ((float) 3.0 * ((float) 15 + (float) 4 * Rnu)) * eps * eps;
	U[hdoti] = (float) (float) 2.0 * C * eps * eps;
	U[G0i] = U[G1i] = U[G2i] = U[F2i] = (float) 0.0;
	for (int l = 3; l < LMAX; l++) {
		U[FLi + l] = (float) 0.0;
		U[NLi + l] = (float) 0.0;
		U[GLi + l] = (float) 0.0;
	}
	U[etai] =
			((float) 0.5 * U[hdoti]
					- ((float) 1.5 * (Oc * U[deltaci] + Ob * U[deltabi])
							+ (float) 1.5 * (Ogam * U[deltagami] + Onu * U[deltanui]))) / (eps * eps);
}

__global__ void einstein_boltzman_kernel(cos_state* states, const zero_order_universe* uni_ptr, float* ks, float amin,
		float amax, float norm, int nmax, bool cont) {
	const int& tid = threadIdx.x;
	const int& bid = blockIdx.x;
	const int& bsz = blockDim.x;
	const int myindex = bid * bsz + tid;
	if( myindex < nmax) {
		if( !cont) {
			einstein_boltzmann_init(states + myindex, uni_ptr, ks[myindex], norm, amin);
		}
		einstein_boltzmann(states + myindex, uni_ptr, ks[myindex], amin, amax);
	}
}

__device__
void einstein_boltzmann(cos_state* uptr, const zero_order_universe *uni_ptr, float k, float amin, float amax) {
	const auto &uni = *uni_ptr;
	if (amin < uni.amin && amax > uni.amax) {
		printf("out of range error in einstein_boltzmann\n");
	}
	cos_state& U = *uptr;
	cos_state U0;
	float loga = logf(amin);
	float logamax = logf(amax);
	float omega_m = uni.params.omega_b + uni.params.omega_c;
	float omega_r = uni.params.omega_gam + uni.params.omega_nu;
	while (loga < logamax) {
		float Oc, Ob, Ogam, Onu, Or;
		float a = expf(loga);
		float hubble = hubble_function(a, uni.params.hubble, omega_m, omega_r);
		float eps = k / (a * hubble);
		uni.compute_radiation_fractions(Ogam, Onu, a);
		uni.compute_matter_fractions(Oc, Ob, a);
		Or = Ogam + Onu;
		float cs2 = uni.cs2(a);
		float lambda_i = 0.0;
		lambda_i = max(lambda_i, sqrtf(((float) LMAX + (float) 1.0) / ((float) LMAX + (float) 3.0)) * eps);
		lambda_i = max(lambda_i,
				sqrtf((float) 3.0 * powf(eps, 4) + (float) 8.0 * eps * eps * Or) / sqrtf((float) 5) / eps);
		float lambda_r = (eps + sqrtf(eps * eps + (float) 4.0 * cs2 * powf(eps, (float) 4))) / ((float) 2.0 * eps);
		float dloga_i = (float) 2.0 * (float) 1.73 / lambda_i;
		float dloga_r = (float) 2.0 * (float) 2.51 / lambda_r;
		float dloga = min(min((float) 5e-2, min((float) 0.9 * dloga_i, (float) 0.1 * dloga_r)), logamax - loga);
		float loga0 = loga;
		const auto compute_expflicit =
				[&](int step) {
					U0 = U;
					cos_state dudt;
					constexpr float beta[3] = {1, 0.25, (2.0 / 3.0)};
					constexpr float tm[3] = {0, 1, 0.5};
					for (int i = 0; i < 3; i++) {
						loga = loga0 + (float) 0.5 * (tm[i] + step) * dloga;
						a = expf(loga);
						hubble = hubble_function(a, uni.params.hubble, omega_m,
								omega_r);
						eps = k / (a * hubble);
						uni.compute_radiation_fractions(Ogam,Onu,a);
						uni.compute_matter_fractions(Oc,Ob,a);
						Or = Ogam + Onu;
						cs2 = uni.cs2(a);
						dudt[taui] = (float) 1.0 / (a * hubble);
						dudt[etai] = ((float) 1.5 * ((Ob * U[thetabi]) + ((float) 4.0 / (float) 3.0) * (Ogam * U[thetagami] + Onu * U[thetanui])) / eps);
						float factor = ((a * omega_m) + (float) 4 * a * a * a * a * ((float) 1 - omega_m - omega_r))
						/ ((float) 2 * a * omega_m + (float) 2 * omega_r + (float) 2 * a * a * a * a * ((float) 1 - omega_m - omega_r));
						dudt[hdoti] =
						(-factor * U[hdoti] - ((float) 3.0 * (Oc * U[deltaci] + Ob * U[deltabi]) + (float) 6.0 * (Ogam * U[deltagami] + Onu * U[deltanui])));
						dudt[deltaci] = -(float) 0.5 * U[hdoti];
						dudt[deltabi] = -eps * U[thetabi] - (float) 0.5 * U[hdoti];
						dudt[deltagami] = -(float) 4.0 / (float) 3.0 * eps * U[thetagami] - ((float) 2.0 / (float) 3.0) * U[hdoti];
						dudt[deltanui] = -(float) 4.0 / (float) 3.0 * eps * U[thetanui] - ((float) 2.0 / (float) 3.0) * U[hdoti];
						dudt[thetabi] = -U[thetabi] + cs2 * eps * U[deltabi];
						dudt[thetagami] = eps * ((float) 0.25 * U[deltagami] - (float) 0.5 * U[F2i]);
						dudt[thetanui] = eps * ((float) 0.25 * U[deltanui] - (float) 0.5 * U[N2i]);
						dudt[F2i] = ((float) 8.0 / (float) 15.0) * eps * U[thetagami] + ((float) 4.0 / (float) 15.0) * U[hdoti] + ((float) 8.0 / (float) 5.0) * dudt[etai]
						- ((float) 3.0 / (float) 5.0) * eps * U[FLi + 3];
						dudt[N2i] = ((float) 8.0 / (float) 15.0) * eps * U[thetanui] + ((float) 4.0 / (float) 15.0) * U[hdoti] + ((float) 8.0 / (float) 5.0) * dudt[etai]
						- ((float) 3.0 / (float) 5.0) * eps * U[NLi + 3];
						dudt[GLi + 0] = -eps * U[GLi + 1];
						dudt[GLi + 1] = eps / (float) (3) * (U[GLi + 0] - (float) 2 * U[GLi + 2]);
						dudt[GLi + 2] = eps / (float) (5) * ((float) 2 * U[GLi + 1] - (float) 3 * U[GLi + 3]);
						for (int l = 3; l < LMAX - 1; l++) {
							dudt[FLi + l] = eps / (float) (2 * l + 1) * ((float) l * U[FLi - 1 + l] - (float) (l + 1) * U[FLi + 1 + l]);
							dudt[NLi + l] = eps / (float) (2 * l + 1) * ((float) l * U[NLi - 1 + l] - (float) (l + 1) * U[NLi + 1 + l]);
							dudt[GLi + l] = eps / (float) (2 * l + 1) * ((float) l * U[GLi - 1 + l] - (float) (l + 1) * U[GLi + 1 + l]);
						}
						dudt[FLi + LMAX - 1] = (eps * U[FLi + LMAX - 2]) / (float) (2 * LMAX - 1);
						dudt[NLi + LMAX - 1] = (eps * U[NLi + LMAX - 2]) / (float) (2 * LMAX - 1);
						dudt[GLi + LMAX - 1] = (eps * U[GLi + LMAX - 2]) / (float) (2 * LMAX - 1);
						for (int f = 0; f < NFIELD; f++) {
							U[f] = ((float) 1 - beta[i]) * U0[f] + beta[i] * (U[f] + dudt[f] * dloga * (float) 0.5);
						}
					}
				};

		auto compute_implicit_dudt =
				[&](float loga, float dloga) {
					a = expf(loga);
					float thetab = U[thetabi];
					float thetagam = U[thetagami];
					float F2 = U[F2i];
					float G0 = U[G0i];
					float G1 = U[G1i];
					float G2 = U[G2i];
					float thetab0 = thetab;
					float thetagam0 = thetagam;
					float F20 = F2;
					float G00 = G0;
					float G10 = G1;
					float G20 = G2;
					float sigma = uni.sigma_T(a);

					thetab = -((-(float) 3 * Ob * thetab0 - (float) 3 * dloga * Ob * sigma * thetab0 - (float) 4 * dloga * Ogam * sigma * thetagam0)
							/ ((float) 3 * Ob + (float) 3 * dloga * Ob * sigma + (float) 4 * dloga * Ogam * sigma));
					thetagam = -((-(float) 3 * dloga * Ob * sigma * thetab0 - (float) 3 * Ob * thetagam0 - (float) 4 * dloga * Ogam * sigma * thetagam0)
							/ ((float) 3 * Ob + (float) 3 * dloga * (float) Ob * sigma + (float) 4 * dloga * Ogam * sigma));
					F2 = -((-(float) 10 * F20 - (float) 4 * dloga * F20 * sigma - dloga * G00 * sigma - dloga * G20 * sigma)
							/ (((float) 1 + dloga * sigma) * ((float) 10 + (float) 3 * dloga * sigma)));
					G0 = -((-(float) 10 * G00 - (float) 5 * dloga * F20 * sigma - (float) 8 * dloga * G00 * sigma - (float) 5 * dloga * G20 * sigma)
							/ (((float) 1 + dloga * sigma) * ((float) 10 + (float) 3 * dloga * sigma)));
					G1 = G10 / ((float) 1 + dloga * sigma);
					G2 = -((-(float) 10 * G20 - dloga * F20 * sigma - dloga * G00 * sigma - (float) 4 * dloga * G20 * sigma)
							/ (((float) 1 + dloga * sigma) * ((float) 10 + (float) 3 * dloga * sigma)));
					array<float, NFIELD> dudt;
					for (int f = 0; f < NFIELD; f++) {
						dudt[f] = (float) 0.0;
					}
					dudt[thetabi] = (thetab - thetab0) / dloga;
					dudt[thetagami] = (thetagam - thetagam0) / dloga;
					dudt[F2i] = (F2 - F20) / dloga;
					dudt[G0i] = (G0 - G00) / dloga;
					dudt[G1i] = (G1 - G10) / dloga;
					dudt[G2i] = (G2 - G20) / dloga;
					for (int l = 3; l < LMAX - 1; l++) {
						dudt[GLi + l] = U[GLi + l] * ((float) 1 / ((float) 1 + dloga * sigma) - (float) 1) / dloga;
						dudt[FLi + l] = U[FLi + l] * ((float) 1 / ((float) 1 + dloga * sigma) - (float) 1) / dloga;
					}
					dudt[GLi + LMAX - 1] = U[GLi + LMAX - 1]
					* ((float) 1 / ((float) 1 + (sigma + (float) LMAX / (U[taui] * a * hubble) / ((float) 2 * (float) LMAX - (float) 1))) - (float) 1) / dloga;
					dudt[FLi + LMAX - 1] = U[FLi + LMAX - 1]
					* ((float) 1 / ((float) 1 + (sigma + (float) LMAX / (U[taui] * a * hubble) / ((float) 2 * (float) LMAX - (float) 1))) - (float) 1) / dloga;
					return dudt;
				};

		compute_expflicit(0);
		float gamma = (float) 1.0 - (float) 1.0 / sqrtf((float) 2);

		auto dudt1 = compute_implicit_dudt(loga + gamma * dloga, gamma * dloga);
		for (int f = 0; f < NFIELD; f++) {
			U[f] += dudt1[f] * ((float) 1.0 - (float) 2.0 * gamma) * dloga;
		}
		auto dudt2 = compute_implicit_dudt(loga + ((float) 1.0 - gamma) * dloga, gamma * dloga);
		for (int f = 0; f < NFIELD; f++) {
			U[f] += (dudt1[f] * ((float) -0.5 + (float) 2.0 * gamma) + dudt2[f] * (float) 0.5) * dloga;
		}

		compute_expflicit(1);

		loga = loga0 + dloga;
	}
}

void einstein_boltzmann_interpolation_function(interp_functor<float>* den_k_func, interp_functor<float>* vel_k_func,
		cos_state* U, zero_order_universe* uni, float kmin, float kmax, float norm, int N, float astart, float astop, bool cont) {
	int block_size = 32;
	int nblocks = (N-1) / block_size + 1;
	float dlogk = 1.0e-2;
	float logkmin = logf(kmin) - dlogk;
	float logkmax = logf(kmax) + dlogk;
	dlogk = (logkmax - logkmin) / (float) (N - 1);
	float* ks;
	CUDA_MALLOC(ks, N);
	float littleh = uni->params.hubble;

	for (int i = 0; i < N; i++) {
		ks[i] = expf(logkmin + (float) i * dlogk) * littleh;
	}
	einstein_boltzman_kernel<<<nblocks,block_size>>>(U, uni,ks, astart, astop, norm, N, cont);
	CUDA_CHECK(cudaDeviceSynchronize());

	vector<float> den_k(N), vel_k(N);
	float oc = uni->params.omega_c;
	float ob = uni->params.omega_b;
	float om = oc + ob;
	float h3 = sqr(littleh) * littleh;
	oc /= om;
	ob /= om;
	const auto hubble =
			[uni](float a) {
				return hubble_function(a,uni->params.hubble, uni->params.omega_c + uni->params.omega_b, uni->params.omega_gam + uni->params.omega_nu);
			};
	float H = hubble(astop);
	for (int i = 0; i < N; i++) {
		float k = ks[i];
		float eps = k / (astop * H);
		den_k[i] = sqr(ob * U[i][deltabi] + oc * U[i][deltaci]) * h3;
		vel_k[i] = h3
				* sqr((ob * (eps * U[i][thetabi] + (float) 0.5 * U[i][hdoti]) + oc * ((float) 0.5 * U[i][hdoti])) / eps);
	//	printf("%e %e\n", k, vel_k[i]);
	}
	build_interpolation_function(den_k_func, (den_k), expf(logkmin), expf(logkmax));
	build_interpolation_function(vel_k_func, (vel_k), expf(logkmin), expf(logkmax));
	printf("Matter and Velocity Power Spectrum\n");
	for (int i = 0; i < N; i++) {
//		float k = expf(logkmin + (float) i * dlogk);
	}
	CUDA_FREE(ks);
}

