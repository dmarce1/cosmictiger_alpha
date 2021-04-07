#include <cosmictiger/ewald_indices.hpp>
#include <cosmictiger/interactions.hpp>

#define NREAL 147
#define NFOUR 92

static array<array<float, NDIM>, NREAL> real_indices;
static array<array<float, NDIM>, NFOUR> four_indices;
static array<expansion<float>, NFOUR> four_expanse;

static __constant__ array<array<float, NDIM>, NREAL> real_indices_dev;
static __constant__ array<array<float, NDIM>, NFOUR> four_indices_dev;
static __constant__ array<expansion<float>, NFOUR> four_expanse_dev;

void ewald_const::init() {
	int n2max = 10;
	int nmax = std::sqrt(n2max) + 1;
	array<float, NDIM> this_h;
	int count = 0;
	for (int i = -nmax; i <= nmax; i++) {
		for (int j = -nmax; j <= nmax; j++) {
			for (int k = -nmax; k <= nmax; k++) {
				const int i2 = i * i + j * j + k * k;
				if (i2 <= n2max) {
					this_h[0] = i;
					this_h[1] = j;
					this_h[2] = k;
					real_indices[count++] = this_h;
				}
			}
		}
	}
	printf("nreal = %i\n", count);
	n2max = 8;
	nmax = std::sqrt(n2max) + 1;
	count = 0;
	for (int i = -nmax; i <= nmax; i++) {
		for (int j = -nmax; j <= nmax; j++) {
			for (int k = -nmax; k <= nmax; k++) {
				if (i * i + j * j + k * k <= n2max) {
					this_h[0] = i;
					this_h[1] = j;
					this_h[2] = k;
					const auto hdot = sqr(this_h[0]) + sqr(this_h[1]) + sqr(this_h[2]);
					if (hdot > 0) {
						four_indices[count++] = this_h;
					}
				}
			}
		}
	}
	printf("nfour = %i\n", count);
	count = 0;
	for (int i = 0; i < NFOUR; i++) {
		array<float, NDIM> h = four_indices[i];
		const float h2 = sqr(h[0]) + sqr(h[1]) + sqr(h[2]);                     // 5 OP
		expansion<float> D;
		D = 0.0;
		const float c0 = 1.0 / h2 * exp(-M_PI * M_PI * h2 / 4.0);
		D() = -(1.0 / M_PI) * c0;
		for (int a = 0; a < NDIM; a++) {
			D(a) = 2.0 * h[a] * c0;
			for (int b = 0; b <= a; b++) {
				D(a, b) = 4.0 * M_PI * h[a] * h[b] * c0;
				for (int c = 0; c <= b; c++) {
					D(a, b, c) = -8.0 * M_PI * M_PI * h[a] * h[b] * h[c] * c0;
					for (int d = 0; d <= c; d++) {
						D(a, b, c, d) = -16.0 * M_PI * M_PI * M_PI * h[a] * h[b] * h[c] * h[d] * c0;
					}
				}
			}
		}
		four_expanse[count++] = D;
	}
	CUDA_CHECK(cudaMemcpyToSymbol(real_indices_dev,&real_indices,sizeof(real_indices)));
	CUDA_CHECK(cudaMemcpyToSymbol(four_indices_dev,&four_indices,sizeof(four_indices)));
	CUDA_CHECK(cudaMemcpyToSymbol(four_expanse_dev,&four_expanse,sizeof(four_expanse)));

}

CUDA_EXPORT int ewald_const::nfour() {
	return NFOUR;
}

CUDA_EXPORT int ewald_const::nreal() {
	return NREAL;
}

CUDA_EXPORT const array<float, NDIM>& ewald_const::real_index(int i) {
#ifdef __CUDA_ARCH__
	return real_indices_dev[i];
#else
	return real_indices[i];
#endif
}

CUDA_EXPORT const array<float, NDIM>& ewald_const::four_index(int i) {
#ifdef __CUDA_ARCH__
	return four_indices_dev[i];
#else
	return four_indices[i];
#endif
}

CUDA_EXPORT const expansion<float>& ewald_const::four_expansion(int i) {
#ifdef __CUDA_ARCH__
	return four_expanse_dev[i];
#else
	return four_expanse[i];
#endif
}



/*

D[0] = 2.837291e+00
D[4] = -4.188790e+00
D[7]= -4.188790e+00+
D[9] = -4.188790e+00
D[20] = -7.42e+01
D[23] = 3.73e+01
D[25] = 3.73e+01
D[30] = -7.42e+01
D[32] = 3.73e+01
D[34] = -7.42e+01
*/
