#include <cosmictiger/defs.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/particle.hpp>
#include <cosmictiger/map.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/cosmos.hpp>

int drift_particles(particle_sets partsets, double dt, double a0, double* ekin, double* momx, double* momy,
		double* momz, double tau, double tau_max) {
	const int gsz = 2 * hpx::threads::hardware_concurrency();
	static mutex_type mtx;
	std::vector<hpx::future<void>> futs;
	static std::atomic<int> rc(0);
	rc = 0;
	const auto ddt = cosmos_drift_dtau(a0, dt);
	const auto a = 1.0 / ddt;
	for (int bid = 0; bid < gsz; bid++) {
		auto func = [a,bid, gsz,  dt,ekin, momx, momy, momz, tau, tau_max,&partsets]() {
			for( int pi = 0; pi < NPART_TYPES; pi++) {
				auto& parts = *partsets.sets[pi];
				auto map_ws = get_map_workspace();
				map_ws.wt = partsets.weights[pi];
				const bool map = global().opts.map_size > 0;
				const size_t nparts = parts.size();
				const size_t start = bid * nparts / gsz;
				const size_t stop = (bid + 1) * nparts / gsz;
				const float ainv = 1.0f / a;
				const float dtinv = 1.f / dt;
#ifdef CONFORMAL_TIME
				const float dteff = dt * ainv;
#else
				const float dteff = dt * ainv * ainv;
#endif
				int myrc = 0;
				double myekin, mymomx, mymomy, mymomz;
				myekin = 0;
				mymomx = 0;
				mymomy = 0;
				mymomz = 0;
				array<double,NDIM> x0, x1;
				for (size_t i = start; i < stop; i++) {
					double x = parts.pos(0, i).to_double();
					double y = parts.pos(1, i).to_double();
					double z = parts.pos(2, i).to_double();
					x0[0] = x;
					x0[1] = y;
					x0[2] = z;
					const float vx = parts.vel(0,i);
					const float vy = parts.vel(1,i);
					const float vz = parts.vel(2,i);
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
					x1[0] = x;
					x1[1] = y;
					x1[2] = z;
					if( map ) {
						myrc +=map_add_part(x0, x1, tau, dt,dtinv, tau_max, map_ws);
					}
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
				cleanup_map_workspace(map_ws, tau, dt, tau_max);
				std::lock_guard<mutex_type> lock(mtx);
				*ekin += myekin;
				*momx += mymomx;
				*momy += mymomy;
				*momz += mymomz;
				rc += myrc;
			}
		};
		futs.push_back(hpx::async(func));
	}
	hpx::wait_all(futs.begin(), futs.end());
	return int(rc);
}
