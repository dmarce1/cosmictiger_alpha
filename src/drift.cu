#include <cosmictiger/drift.hpp>
#include <cosmictiger/global.hpp>

CUDA_KERNEL drift_kernel(particle_set parts, double dt, double a0, double a1) {
	const int& tid = threadIdx.x;
	const int& bid = blockIdx.x;
	const int& gsz = gridDim.x;
	const size_t nparts = parts.size();
	const size_t start = bid * nparts / gsz;
	const size_t stop = (bid + 1) * nparts / gsz;
	const float dteff = dt * 0.5 * (1.0 / a0 + 0.5 / a1);
	for (size_t i = start + tid; i < stop; i += DRIFT_BLOCK_SIZE) {
	//	printf( "%li %li %li\n", i - start, start-start, stop-start);
		double x = parts.pos(0, i).to_double();
		double y = parts.pos(1, i).to_double();
		double z = parts.pos(2, i).to_double();
		const double vx = (double) parts.vel(0, i);
		const double vy = (double) parts.vel(1, i);
		const double vz = (double) parts.vel(2, i);
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
		parts.pos(0, i) = x;
		parts.pos(1, i) = y;
		parts.pos(2, i) = z;
	}
//	if( tid == 0 )
//	printf( "Block %i stop\n", bid);
}

void drift_particles(particle_set parts, double dt, double a0, double a1) {
	const int nblock = DRIFT_OCCUPANCY * 2 * global().cuda.devices[0].multiProcessorCount;
	auto stream = get_stream();
	parts.prepare_drift(stream);
	drift_kernel<<<nblock,DRIFT_BLOCK_SIZE,0,stream>>>(parts, dt, a0, a1);
	CUDA_CHECK(cudaStreamSynchronize(stream));
	cleanup_stream(stream);
}
