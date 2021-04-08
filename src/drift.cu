#include <cosmictiger/drift.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/cosmos.hpp>


CUDA_KERNEL drift_kernel(particle_set parts, double dt, double a, double* ekin, double* momx, double* momy,
		double* momz) {
	const int& tid = threadIdx.x;
	const int& bid = blockIdx.x;
	const int& gsz = gridDim.x;
	const size_t nparts = parts.size();
	const size_t start = bid * nparts / gsz;
	const size_t stop = (bid + 1) * nparts / gsz;
	const float ainv = 1.0 / a;
	const float dteff = dt * ainv;
	double myekin, mymomx, mymomy, mymomz;
	myekin = 0;
	mymomx = 0;
	mymomy = 0;
	mymomz = 0;
	for (size_t i = start + tid; i < stop; i += DRIFT_BLOCK_SIZE) {
		//	printf( "%li %li %li\n", i - start, start-start, stop-start);
		double x = parts.pos(0,i).to_double();
		double y = parts.pos(1,i).to_double();
		double z = parts.pos(2,i).to_double();
		const float vx = parts.vel(i).p.x;
		const float vy = parts.vel(i).p.y;
		const float vz = parts.vel(i).p.z;
		const float ux = vx * ainv;
		const float uy = vy * ainv;
		const float uz = vz * ainv;
		myekin += 0.5 * (ux * ux + uy * uy + uz * uz);
		mymomx += ux;
		mymomy += uy;
		mymomz += uz;
		x += vx * dteff;
		y += vy * dteff;
		z += vz * dteff;
		while (x >= 1.0) {
			x -= 1.0;
		}
		while (y >= 1.0) {
			y -= 1.0;
		}
		while (z >= 1.0) {
			z -= 1.0;
		}
		while (x < 0.0) {
			x += 1.0;
		}
		while (y < 0.0) {
			y += 1.0;
		}
		while (z < 0.0) {
			z += 1.0;
		}
		parts.pos(0,i) = x;
		parts.pos(1,i) = y;
		parts.pos(2,i) = z;
	}
	for (int P = warpSize / 2; P >= 1; P /= 2) {
		myekin += __shfl_down_sync(0xFFFFFFFF, myekin, P);
		mymomx += __shfl_down_sync(0xFFFFFFFF, mymomx, P);
		mymomy += __shfl_down_sync(0xFFFFFFFF, mymomy, P);
		mymomz += __shfl_down_sync(0xFFFFFFFF, mymomz, P);
	}
	if (tid % warpSize == 0) {
		atomicAdd(ekin, myekin);
		atomicAdd(momx, mymomx);
		atomicAdd(momy, mymomy);
		atomicAdd(momz, mymomz);
	}
//	if( tid == 0 )
//	printf( "Block %i stop\n", bid);
}

int cpu_drift_kernel(particle_set parts, double dt, double a, double* ekin, double* momx, double* momy, double* momz, double tau, double tau_max);

