#include <cosmictiger/kick_return.hpp>
#include <cosmictiger/hpx.hpp>

static kick_return cpu_return;
static mutex_type mtx;

void kick_return_init_gpu(int min_rung);
kick_return kick_return_get_gpu();

void kick_return_init(int min_rung) {
	cpu_return.min_rung = min_rung;
	for (int i = 0; i < MAX_RUNG; i++) {
		cpu_return.rung_cnt[i] = 0;
	}
	kick_return_init_gpu(min_rung);
}

kick_return kick_return_get() {
	auto rc = kick_return_get_gpu();
	for (int i = 0; i < MAX_RUNG; i++) {
		rc.rung_cnt[i] += cpu_return.rung_cnt[i];
	}
	return rc;
}

void kick_return_show() {
	kick_return rc;
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
	printf("Rungs ");
	for (int i = min_rung; i < max_rung; i++) {
		printf("%8i ", i);
	}
	printf("\n      ");
	for (int i = min_rung; i < max_rung; i++) {
		printf("%8i ", rc.rung_cnt[i]);
	}
	printf( "\n");

}

void kick_return_update_rung_cpu(int rung) {
	std::lock_guard<mutex_type> lock(mtx);
	cpu_return.rung_cnt[rung]++;
}
