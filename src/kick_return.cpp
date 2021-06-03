#include <cosmictiger/kick_return.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/timer.hpp>

static kick_return cpu_return;
static mutex_type mtx;

static timer tm;

void kick_return_init_gpu(int min_rung);
kick_return kick_return_get_gpu();

HPX_PLAIN_ACTION(kick_return_init);
HPX_PLAIN_ACTION(kick_return_get);

void kick_return_init(int min_rung) {
	std::vector<hpx::future<void>> futs;
	if (hpx_rank() == 0) {
		for (int i = 1; i < hpx_size(); i++) {
			futs.push_back(hpx::async<kick_return_init_action>(hpx_localities()[i], min_rung));
		}
	}
	cpu_return.min_rung = min_rung;
	for (int i = 0; i < MAX_RUNG; i++) {
		cpu_return.rung_cnt[i] = 0;
	}
	for (int i = 0; i < KR_COUNT; i++) {
		cpu_return.flop[i] = 0;
		cpu_return.count[i] = 0;
	}
	cpu_return.phis = 0.f;
	for (int dim = 0; dim < NDIM; dim++) {
		cpu_return.forces[dim] = 0.f;
	}
	kick_return_init_gpu(min_rung);
	tm.start();
	hpx::wait_all(futs.begin(), futs.end());
}

kick_return kick_return_get() {
	std::vector<hpx::future<kick_return>> futs;
	if (hpx_rank() == 0) {
		for (int i = 1; i < hpx_size(); i++) {
			futs.push_back(hpx::async<kick_return_get_action>(hpx_localities()[i]));
		}
	}
	kick_return rc = kick_return_get_gpu();
	rc.phis += cpu_return.phis;
	for (int dim = 0; dim < NDIM; dim++) {
		rc.forces[dim] += cpu_return.forces[dim];
	}
	for (int i = 0; i < MAX_RUNG; i++) {
		rc.rung_cnt[i] += cpu_return.rung_cnt[i];
	}
	for (int i = 0; i < KR_COUNT; i++) {
		rc.count[i] += cpu_return.count[i];
		rc.flop[i] += cpu_return.flop[i];
	}
	if (hpx_rank() == 0) {
		for (auto& f : futs) {
			const auto tmp = f.get();
			rc.phis += tmp.phis;
			for (int dim = 0; dim < NDIM; dim++) {
				rc.forces[dim] += tmp.forces[dim];
			}
			for (int i = 0; i < MAX_RUNG; i++) {
				rc.rung_cnt[i] += tmp.rung_cnt[i];
			}
			for (int i = 0; i < KR_COUNT; i++) {
				rc.count[i] += tmp.count[i];
				rc.flop[i] += tmp.flop[i];
			}
		}
	}
	return rc;
}

void kick_return_update_interactions_cpu(int itype, int count, int flops) {
	std::lock_guard<mutex_type> lock(mtx);
	cpu_return.flop[itype] += flops;
	cpu_return.count[itype] += count;
}

int kick_return_max_rung() {
	kick_return rc = kick_return_get();
	int max_rung = 0;
	for (int i = 0; i < MAX_RUNG; i++) {
		if (rc.rung_cnt[i]) {
			max_rung = std::max(max_rung, i);
		}
	}
	return max_rung;
}

void kick_return_show() {
	kick_return rc;
	tm.stop();
	rc = kick_return_get();
	int min_rung = -1;
	int max_rung = 0;
	for (int i = 0; i < MAX_RUNG; i++) {
		if (rc.rung_cnt[i]) {
			max_rung = std::max(max_rung, i);
			if (min_rung == -1) {
				min_rung = i;
			}
		}
	}
	const auto elapsed = tm.read();
	tm.reset();
	PRINT("\nKick took %e seconds\n\n", elapsed);
	float phi_sum = 0.f;
	const auto fac = 1.0 / global().opts.nparts;
	PRINT("Potential Energy / Total Forces    = %e %e %e %e\n", rc.phis * fac, rc.forces[0] * fac, rc.forces[1] * fac,
			rc.forces[2] * fac);
	PRINT("Rungs ");
	for (int i = min_rung; i <= max_rung; i++) {
		PRINT("%8i ", i);
	}
	PRINT("\n      ");
	for (int i = min_rung; i <= max_rung; i++) {
		PRINT("%8i ", rc.rung_cnt[i]);
	}
	PRINT("\n");

	PRINT("Interactions / part : GFLOP\n");
	array<double, KR_COUNT> count_pct, flop_pct, flop_per;
	double flop_tot = 0, count_tot = 0;
	for (int i = 0; i < KR_COUNT; i++) {
		flop_per[i] = rc.flop[i] / rc.count[i];
		rc.count[i] /= global().opts.nparts;
		flop_tot += rc.flop[i];
		count_tot += rc.count[i];
	}
	for (int i = 0; i < KR_COUNT; i++) {
		count_pct[i] = rc.count[i] / count_tot * 100.0;
		flop_pct[i] = rc.flop[i] / flop_tot * 100.0;
	}
	const char* names[] = { "PP", "PC", "CP", "CC", "OP", "EW" };
	const auto fac1 = 1.0 / 1024.0 / 1024.0 / 1024.0;
	PRINT("Evals per particles | FLOP | FLOP per eval\n");
	for (int i = 0; i < 6; i++) {
		PRINT("%4s   : %8.3e (%5.2f%%)  %8.3e (%5.2f%%) %.2f\n", names[i], rc.count[i], count_pct[i], rc.flop[i] * fac1,
				flop_pct[i], flop_per[i]);
	}
	PRINT("\nTotal GFLOP  = %8.3e\n", flop_tot * fac1);
	PRINT("Total GFLOPS = %8.3e\n\n", flop_tot * fac1 / elapsed);

}

void kick_return_update_pot_cpu(float phi, float fx, float fy, float fz) {
	std::lock_guard<mutex_type> lock(mtx);
	cpu_return.phis += phi;
	cpu_return.forces[0] += fx;
	cpu_return.forces[1] += fx;
	cpu_return.forces[2] += fx;
}

void kick_return_update_rung_cpu(int rung) {
	if (rung >= MAX_RUNG) {
		printf("Rung out of range ! %i\n", rung);
	} else {
		std::lock_guard<mutex_type> lock(mtx);
		cpu_return.rung_cnt[rung]++;
	}

}
