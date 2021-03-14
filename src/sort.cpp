#include <cosmictiger/sort.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/lockfree_queue.hpp>

struct sort_work {
	std::pair<size_t, size_t> parts;
	int xdim;
	double xmid;
	std::shared_ptr<hpx::promise<size_t>> promise;
};

lockfree_queue<sort_work, SORT_QUEUE_SIZE> sort_queue;

std::atomic<int> sort_daemon_running(0);

std::atomic<int> stop_daemon(0);

void stop_sort_daemon() {
	stop_daemon++;
}

void sort_daemon(particle_set parts) {
	static bool first = true;
	using entry_type = std::pair<std::function < bool(size_t*)>,std::shared_ptr<hpx::promise<size_t>>>;
	static std::vector<entry_type> completions;

	while (sort_queue.size()) {
		//	printf( "Popping work\n");
		auto tmp = sort_queue.pop();
		auto func = sort_particles(parts, tmp.parts.first, tmp.parts.second, tmp.xmid, tmp.xdim);
		completions.push_back(std::make_pair(func, tmp.promise));
	}

	int i = 0;
	while (i < completions.size()) {
		size_t pmid;
		if (completions[i].first(&pmid)) {
			completions[i].second->set_value(pmid);
			completions[i] = completions.back();
			completions.pop_back();
		} else {
			i++;
		}
	}
	if (stop_daemon == 0) {
		hpx::async(sort_daemon, parts);
	} else {
		sort_daemon_running = 0;
		stop_daemon = 0;
	}
}

hpx::future<size_t> send_sort(particle_set parts, size_t begin, size_t end, double xmid, int xdim) {
	sort_work sw;
	sw.parts.first = begin;
	sw.parts.second = end;
	sw.xmid = xmid;
	sw.xdim = xdim;
	sw.promise = std::make_shared<hpx::promise<size_t>>();
	auto fut = sw.promise->get_future();
//	printf( "Pushing work\n");
	sort_queue.push(std::move(sw));
	if (sort_daemon_running++ == 0) {
		hpx::async(sort_daemon, parts);
	}
	return fut;
}

