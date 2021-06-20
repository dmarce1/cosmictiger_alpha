#include <cosmictiger/defs.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/vector.hpp>
#include <cosmictiger/map.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/math.hpp>

void hpx_yield();

/*************** MOST OF THESE ROUTINES ADAPTED FROM THE C HEALPIX LIBRARY *****************************/

__constant__ float twothird = 2.0 / 3.0;
__constant__ float pi = 3.141592653589793238462643383279502884197;
__constant__ float twopi = 6.283185307179586476925286766559005768394;
__constant__ float halfpi = 1.570796326794896619231321691639751442099;
__constant__ float inv_halfpi = 0.6366197723675813430755350534900574;

__device__
inline float fmodulo(float v1, float v2);
__device__
inline int imodulo(int v1, int v2);
__device__
int ang2pix_ring_z_phi(int nside_, float z, float phi);
__device__
void vec2pix_ring(int nside, const float *vec, int *ipix);
__global__
void healpix_kernel(const float * __restrict__ x, const float * __restrict__ y, const float * __restrict__ z,
		const float * __restrict__ vx, const float * __restrict__ vy, const float * __restrict__ vz, float taui,
		float tau0, float dtau, float * __restrict__ map, int Npts, int Nside);

void healpix2_map(const vector<float>& x, const vector<float>& y, const vector<float>& z, const vector<float>& vx,
		const vector<float>& vy, const vector<float>& vz, float taui, float tau0, float dtau, map_type map, int Nside) {
	auto stream = get_stream();
	cuda_set_device();
	//cudaFuncAttributes attribs;
//	CUDA_CHECK(cudaFuncGetAttributes(&attribs, healpix_kernel));
	int num_threads = 96;
	int num_blocks;
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, healpix_kernel, num_threads, 0));
	//PRINT( "Healpix occupancy = %i numregs = %i\n", num_blocks, attribs.numRegs);
	num_blocks *= global().cuda.devices[0].multiProcessorCount;

	CUDA_CHECK(cudaMemPrefetchAsync(x.data(), sizeof(float) * x.size(), cuda_device(), stream));
	CUDA_CHECK(cudaMemPrefetchAsync(y.data(), sizeof(float) * y.size(), cuda_device(), stream));
	CUDA_CHECK(cudaMemPrefetchAsync(z.data(), sizeof(float) * z.size(), cuda_device(), stream));
	CUDA_CHECK(cudaMemPrefetchAsync(vx.data(), sizeof(float) * vx.size(), cuda_device(), stream));
	CUDA_CHECK(cudaMemPrefetchAsync(vy.data(), sizeof(float) * vy.size(), cuda_device(), stream));
	CUDA_CHECK(cudaMemPrefetchAsync(vz.data(), sizeof(float) * vz.size(), cuda_device(), stream));
	healpix_kernel<<<num_blocks,num_threads,0,stream>>>(x.data(),y.data(),z.data(),vx.data(),vy.data(),vz.data(), taui, tau0, dtau,*map,x.size(),Nside);
	while (cudaStreamSynchronize(stream) != cudaSuccess) {
		hpx_yield();
	}
	CUDA_CHECK(cudaStreamSynchronize(stream));
	cleanup_stream(stream);
}

__global__
void healpix_kernel(const float * __restrict__ x, const float * __restrict__ y, const float * __restrict__ z,
		const float * __restrict__ vx, const float * __restrict__ vy, const float * __restrict__ vz, float taui,
		float tau0, float dtau, float * __restrict__ map, int Npts, int Nside) {
	const int tid = threadIdx.x;
	const int bsz = blockDim.x;
	const int bid = blockIdx.x;
	const int gsz = gridDim.x;
	const int start = size_t(bid) * size_t(Npts) / size_t(gsz);
	const int stop = size_t(bid + 1) * size_t(Npts) / size_t(gsz);
	const float sqrtauimtau0 = sqr(taui - tau0);
	const float tau0mtaui = tau0 - taui;
	float vec[NDIM];
	int ipix;
	float mag;
	for (int i = start + tid; i < stop; i += bsz) {
		const float u2 = sqr(vx[i], vy[i], vz[i]);
		const float x2 = sqr(x[i], y[i], z[i]);
		const float udotx = vx[i] * x[i] + vy[i] * y[i] + vz[i] * z[i];
		const float A = 1.f - u2;
		const float B = 2.f * (tau0mtaui - udotx);
		const float C = sqrtauimtau0 - x2;
		const float t = -(B + sqrtf(B * B - 4.f * A * C)) / (2.f * A);
		const float x1 = x[i] + vx[i] * t;
		const float y1 = y[i] + vy[i] * t;
		const float z1 = z[i] + vz[i] * t;
		if (sqr(x1, y1, z1) <= 1.f) {
			mag = 1.f / sqr(x1, y1, z1);
			vec[0] = x1;
			vec[1] = y1;
			vec[2] = z1;
			vec2pix_ring(Nside, vec, &ipix);
			atomicAdd(map + ipix, mag);
		}
	}
}

__device__
inline float fmodulo(float v1, float v2) {
	if (v1 >= 0) {
		return (v1 < v2) ? v1 : fmodf(v1, v2);
	} else {
		float tmp = fmodf(v1, v2) + v2;
		return (tmp == v2) ? 0. : tmp;
	}
}

/*! Returns the remainder of the division \a v1/v2.
 The result is non-negative.
 \a v1 can be positive or negative; \a v2 must be positive. */
__device__
inline int imodulo(int v1, int v2) {
	int v = v1 % v2;
	return (v >= 0) ? v : v + v2;
}

__device__
int ang2pix_ring_z_phi(int nside_, float z, float phi) {
	float za = fabsf(z);
	float tt = fmodulo(phi, twopi) * inv_halfpi; /* in [0,4) */

	if (za <= twothird) /* Equatorial region */
	{
		float temp1 = nside_ * (0.5 + tt);
		float temp2 = nside_ * z * 0.75;
		int jp = (int) (temp1 - temp2); /* index of  ascending edge line */
		int jm = (int) (temp1 + temp2); /* index of descending edge line */

		/* ring number counted from z=2/3 */
		int ir = nside_ + 1 + jp - jm; /* in {1,2n+1} */
		int kshift = 1 - (ir & 1); /* kshift=1 if ir even, 0 otherwise */

		int ip = (jp + jm - nside_ + kshift + 1) / 2; /* in {0,4n-1} */
		ip = imodulo(ip, 4 * nside_);

		return nside_ * (nside_ - 1) * 2 + (ir - 1) * 4 * nside_ + ip;
	} else /* North & South polar caps */
	{
		float tp = tt - (int) (tt);
		float tmp = nside_ * sqrt(3 * (1 - za));

		int jp = (int) (tp * tmp); /* increasing edge line index */
		int jm = (int) ((1.0 - tp) * tmp); /* decreasing edge line index */

		int ir = jp + jm + 1; /* ring number counted from the closest pole */
		int ip = (int) (tt * ir); /* in {0,4*ir-1} */
		ip = imodulo(ip, 4 * ir);

		if (z > 0)
			return 2 * ir * (ir - 1) + ip;
		else
			return 12 * nside_ * nside_ - 2 * ir * (ir + 1) + ip;
	}
}

__device__
void vec2pix_ring(int nside, const float *vec, int *ipix) {
	float vlen = sqrtf(fmaf(vec[0], vec[0], fmaf(vec[1], vec[1], vec[2] * vec[2])));
	*ipix = ang2pix_ring_z_phi(nside, vec[2] / vlen, atan2(vec[1], vec[0]));
}

