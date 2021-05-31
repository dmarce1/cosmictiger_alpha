#include <cosmictiger/defs.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/particle_server.hpp>
#include <cosmictiger/map.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/cosmos.hpp>
#include <cosmictiger/drift.hpp>

HPX_PLAIN_ACTION (drift);

int drift_particles(particle_set parts, double dt, double a0, double* ekin, double* momx, double* momy, double* momz,
		double tau, double tau_max) {
	*ekin = *momx = *momy = *momz = 0.0;
	const int gsz = 2 * hpx::thread::hardware_concurrency();
	static mutex_type mtx;
	std::vector<hpx::future<void>> futs;
	static std::atomic<int> rc(0);
	rc = 0;
	const auto ddt = cosmos_drift_dtau(a0, dt);
	double a;
	if (ddt > 0.) {
		a = 1.0 / ddt;
	} else {
		a = 1.0;
	}
	for (int bid = 0; bid < gsz; bid++) {
		auto func = [a,bid, gsz, dt,ekin, momx, momy, momz, tau, tau_max,&parts]() {
			auto map_ws = get_map_workspace();
			const bool map = global().opts.map_size > 0;
			const part_int nparts = parts.size();
			const part_int start = (size_t)bid * (size_t)nparts / (size_t)gsz;
			const part_int stop = (size_t)(bid + 1) * (size_t)nparts / (size_t)gsz;
			const float ainv = 1.0f / a;
			const float dtinv = 1.f / dt;
			const float dteff = dt * ainv;
			int myrc = 0;
			double myekin, mymomx, mymomy, mymomz;
			myekin = 0;
			mymomx = 0;
			mymomy = 0;
			mymomz = 0;
			array<double,NDIM> x0, x1;
			for (part_int i = start; i < stop; i++) {
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
		};
		futs.push_back(hpx::async(func));
	}
	hpx::wait_all(futs.begin(), futs.end());
	return int(rc);
}

drift_return drift(double dt, double a, double tau, double tau_max) {
	std::vector<hpx::future<drift_return>> futs;
	if (hpx_rank() == 0) {
		for (int i = 1; i < hpx_size(); i++) {
			futs.push_back(hpx::async<drift_action>(hpx_localities()[i], dt, a, tau, tau_max));
		}
	}
	particle_server pserv;
	const auto parts = pserv.get_particle_set();
	drift_return rc;
	rc.ekin = rc.momx = rc.momy = rc.momz = 0.0;
	rc.map_cnt = drift_particles(parts, dt, a, &rc.ekin, &rc.momx, &rc.momy, &rc.momz, tau, tau_max);
	if (hpx_rank() == 0) {
		for (auto& f : futs) {
			const auto tmp = f.get();
			rc.ekin += tmp.ekin;
			rc.momx += tmp.momx;
			rc.momy += tmp.momy;
			rc.momz += tmp.momz;
			rc.map_cnt += tmp.map_cnt;
		}
	}
	return rc;
}
