#include <cosmictiger/defs.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/particle.hpp>

void cpu_drift_kernel(particle_set parts, double dt, double a, double* ekin, double* momx, double* momy, double* momz) {
	const int gsz = 2 * hpx::threads::hardware_concurrency();

	static mutex_type mtx;
	std::vector<hpx::future<void>> futs;

	for (int bid = 0; bid < gsz; bid++) {
		auto func = [bid, gsz, &parts, dt,a,ekin, momx, momy, momz]() {
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
			for (size_t i = start; i < stop; i++) {
				double x = parts.pos(0, i).to_double();
				double y = parts.pos(1, i).to_double();
				double z = parts.pos(2, i).to_double();
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
				parts.pos(0, i) = x;
				parts.pos(1, i) = y;
				parts.pos(2, i) = z;
			}
			std::lock_guard<mutex_type> lock(mtx);
			*ekin += myekin;
			*momx += mymomx;
			*momy += mymomy;
			*momz += mymomz;
		};
		futs.push_back(hpx::async(func));
	}
	hpx::wait_all(futs.begin(), futs.end());

}

