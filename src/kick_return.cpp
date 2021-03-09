#include <cosmictiger/kick_return.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/timer.hpp>

static kick_return cpu_return;
static mutex_type mtx;

static timer tm;

void kick_return_init_gpu(int min_rung);
kick_return kick_return_get_gpu();

void kick_return_init(int min_rung) {
	cpu_return.min_rung = min_rung;
	for (int i = 0; i < MAX_RUNG; i++) {
		cpu_return.rung_cnt[i] = 0;
	}
	for (int i = 0; i < KR_COUNT; i++) {
		cpu_return.flop[i] = 0;
		cpu_return.count[i] = 0;
	}
	kick_return_init_gpu(min_rung);
	tm.start();
}

kick_return kick_return_get() {
	auto rc = kick_return_get_gpu();
	for (int i = 0; i < MAX_RUNG; i++) {
		rc.rung_cnt[i] += cpu_return.rung_cnt[i];
	}
	for( int i = 0; i < KR_COUNT; i++) {
		rc.count[i] += cpu_return.count[i];
		rc.flop[i] += cpu_return.flop[i];
	}
	return rc;
}


void kick_return_update_interactions_cpu(int itype, int count, int flops) {
	std::lock_guard<mutex_type> lock(mtx);
	cpu_return.flop[itype] += flops;
	cpu_return.count[itype] +=  count;
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
	printf( "\nKick took %e seconds\n\n", elapsed);
	printf("Rungs ");
	for (int i = min_rung; i < max_rung; i++) {
		printf("%8i ", i);
	}
	printf("\n      ");
	for (int i = min_rung; i < max_rung; i++) {
		printf("%8i ", rc.rung_cnt[i]);
	}
	printf("\n");

	printf( "Interactions / part : GFLOP\n");
	array<double,KR_COUNT> count_pct, flop_pct;
	double flop_tot = 0, count_tot = 0;
	for( int i = 0; i < KR_COUNT; i++) {
		rc.count[i] /= global().opts.nparts;
		flop_tot += rc.flop[i];
		count_tot += rc.count[i];
	}
	for( int i = 0; i < KR_COUNT; i++) {
		count_pct[i] = rc.count[i] / count_tot * 100.0;
		flop_pct[i] = rc.flop[i] / flop_tot * 100.0;
	}

	const auto fac = 1.0 / 1024.0/1024.0/1024.0;
	printf( "PP   : %8.3e (%5.2f%%)  %8.3e (%5.2f%%)\n", rc.count[KR_PP],count_pct[KR_PP], rc.flop[KR_PP]*fac, flop_pct[KR_PP]);
	printf( "PC   : %8.3e (%5.2f%%)  %8.3e (%5.2f%%)\n", rc.count[KR_PC],count_pct[KR_PC], rc.flop[KR_PC]*fac, flop_pct[KR_PC]);
	printf( "CP   : %8.3e (%5.2f%%)  %8.3e (%5.2f%%)\n", rc.count[KR_CP],count_pct[KR_CP], rc.flop[KR_CP]*fac, flop_pct[KR_CP]);
	printf( "CC   : %8.3e (%5.2f%%)  %8.3e (%5.2f%%)\n", rc.count[KR_CC],count_pct[KR_CC], rc.flop[KR_CC]*fac, flop_pct[KR_CC]);
	printf( "CCEW : %8.3e (%5.2f%%)  %8.3e (%5.2f%%)\n", rc.count[KR_EWCC],count_pct[KR_EWCC], rc.flop[KR_EWCC]*fac, flop_pct[KR_EWCC]);
	printf( "OP   : %8.3e (%5.2f%%)  %8.3e (%5.2f%%)\n", rc.count[KR_OP],count_pct[KR_OP], rc.flop[KR_OP]*fac, flop_pct[KR_OP]);
	printf( "\nTotal GFLOP  = %8.3e\n", flop_tot * fac);
	printf( "Total GFLOPS = %8.3e\n\n", flop_tot * fac / elapsed);

}

void kick_return_update_rung_cpu(int rung) {
	std::lock_guard<mutex_type> lock(mtx);
	cpu_return.rung_cnt[rung]++;
}
