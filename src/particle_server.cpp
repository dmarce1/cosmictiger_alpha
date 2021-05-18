#include <cosmictiger/particle_server.hpp>
#include <cosmictiger/global.hpp>

particle_set* particle_server::parts = nullptr;
std::vector<part_int> particle_server::free_indices;
particle_send_type particle_server::part_sends;
domain_bounds particle_server::dbounds;
spinlock_type particle_server::mutex;
std::array<mutex_type, PARTICLE_CACHE_SIZE> particle_server::mutexes;
std::array<particle_cache_type, PARTICLE_CACHE_SIZE> particle_server::caches;

HPX_PLAIN_ACTION(particle_server::init, particle_server_init_action);
HPX_PLAIN_ACTION(particle_server::domain_decomp_gather, particle_server_domain_decomp_gather_action);
HPX_PLAIN_ACTION(particle_server::domain_decomp_send, particle_server_domain_decomp_send_action);
HPX_PLAIN_ACTION(particle_server::domain_decomp_finish, particle_server_domain_decomp_finish_action);
HPX_PLAIN_ACTION(particle_server::domain_decomp_transmit, particle_server_domain_decomp_transmit_action);
HPX_PLAIN_ACTION(particle_server::generate_random, particle_server_generate_random_action);
HPX_PLAIN_ACTION(particle_server::check_domain_bounds, particle_server_check_domain_bounds_action);
HPX_PLAIN_ACTION(particle_server::fetch_cache_line, particle_server_fetch_cache_line_action);

part_int index_to_cache_line(part_int i) {
	return i - i % PARTICLE_CACHE_LINE_SIZE;
}

void particle_server::free_cache() {
	for (int i = 0; i < PARTICLE_CACHE_SIZE; i++) {
		caches[i] = particle_cache_type();
	}
}

void particle_server::read_positions(std::array<std::vector<fixed32>, NDIM>& X, int rank, part_iters rng) {
	if (rank == hpx_rank()) {
		for (part_int i = rng.first; i < rng.second; i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				X[dim].push_back(parts->pos(dim, i));
			}
		}
	} else {
		if (rng.second - rng.first == 0) {
			return;
		}
		global_part_iter piter;
		piter.rank = rank;
		piter.index = rng.first;
		load_cache_line(piter);
		part_int i = rng.first;
		do {
			global_part_iter_hash_lo hashlo;
			piter.index = index_to_cache_line(piter.index);
			const auto loindex = hashlo(piter);
			auto& mutex = mutexes[loindex];
			std::unique_lock<mutex_type> lock(mutexes[loindex]);
			auto& cache = caches[loindex];
			auto& entry = cache[piter];
			do {
				for (int dim = 0; dim < NDIM; dim++) {
					const auto value = entry->X[i - piter.index][dim];
					X[dim].push_back(value);
				}
				i++;
			} while (i % PARTICLE_CACHE_LINE_SIZE != 0 && i != rng.second);
			lock.unlock();
			if (i != rng.second) {
				piter.index = i;
				load_cache_line(piter);
			}
		} while (i != rng.second);
	}
}

void particle_server::load_cache_line(global_part_iter piter) {
	global_part_iter_hash_lo hashlo;
	global_part_iter line_ptr;
	line_ptr.rank = piter.rank;
	line_ptr.index = index_to_cache_line(piter.index);
	const auto loindex = hashlo(line_ptr);
	std::unique_lock<mutex_type> lock(mutexes[loindex]);
	auto& cache = caches[loindex];
	auto i = cache.find(line_ptr);
	if (i == cache.end()) {
		auto& entry = cache[line_ptr];
		auto prms = std::make_shared<hpx::lcos::local::promise<void>>();
		entry = std::make_shared<pos_cache_entry>();
		entry->ready_fut = prms->get_future();
		lock.unlock();
		hpx::apply([prms,i,loindex,line_ptr]() {
			particle_server_fetch_cache_line_action act;
			auto line = act(hpx_localities()[line_ptr.rank], line_ptr.index);
			auto& mutex = mutexes[loindex];
			std::lock_guard<mutex_type> lock(mutex);
			auto& cache = caches[loindex];
			cache[line_ptr]->X = std::move(line);
			prms->set_value();
		});
		lock.lock();
		i = cache.find(line_ptr);
	}
	auto& entry = i->second;
	lock.unlock();
	entry->ready_fut.get();
}

pos_data_t particle_server::fetch_cache_line(part_int index) {
	pos_data_t data;
	data.reserve(PARTICLE_CACHE_LINE_SIZE);
	part_int start = index_to_cache_line(index);
	part_int stop = std::min(start + PARTICLE_CACHE_LINE_SIZE, parts->size());
	for (part_int i = start; i < stop; i++) {
		std::array<fixed32, NDIM> X;
		for (int dim = 0; dim < NDIM; dim++) {
			X[dim] = parts->pos(dim, i);
		}
		data.push_back(X);
	}
	return data;
}

void particle_server::check_domain_bounds() {
	std::vector<hpx::future<void>> futs;
	if (hpx_rank() == 0) {
		for (int i = 1; i < hpx_size(); i++) {
			futs.push_back(hpx::async<particle_server_check_domain_bounds_action>(hpx_localities()[i]));
		}
	}
	auto myrange = dbounds.find_proc_range(hpx_rank());
	for (part_int i = 0; i < parts->size(); i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			const auto x = parts->pos(dim, i).to_double();
			if (x < myrange.begin[dim] || x > myrange.end[dim]) {
				ERROR()
				;
			}
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void particle_server::generate_random() {
	std::vector<hpx::future<void>> futs;
	if (hpx_rank() == 0) {
		for (int i = 1; i < hpx_size(); i++) {
			futs.push_back(hpx::async<particle_server_generate_random_action>(hpx_localities()[i]));
		}
	}

	parts->generate_random(hpx_rank() + 24);

	hpx::wait_all(futs.begin(), futs.end());
}

void particle_server::domain_decomp_send() {
	std::vector<hpx::future<void>> futs;
	if (hpx_rank() == 0) {
		for (int i = 1; i < hpx_size(); i++) {
			futs.push_back(hpx::async<particle_server_domain_decomp_send_action>(hpx_localities()[i]));
		}
	}
	for (int i = 0; i < hpx_size(); i++) {
		if (part_sends[i].parts.size()) {
			futs.push_back(
					hpx::async<particle_server_domain_decomp_transmit_action>(hpx_localities()[i],
							std::move(part_sends[i].parts)));
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void particle_server::apply_domain_decomp() {
	bool complete;
	do {
		complete = domain_decomp_gather();
		domain_decomp_send();
		domain_decomp_finish();
	} while (!complete);
}

void particle_server::domain_decomp_finish() {
	std::vector<hpx::future<void>> futs;
	if (hpx_rank() == 0) {
		for (int i = 1; i < hpx_size(); i++) {
			futs.push_back(hpx::async<particle_server_domain_decomp_finish_action>(hpx_localities()[i]));
		}
	}
	parts->free_particles(free_indices);
	free_indices = decltype(free_indices)();
	part_sends = decltype(part_sends)();
	hpx::wait_all(futs.begin(), futs.end());
}

void particle_server::domain_decomp_transmit(std::vector<particle> new_parts) {
	while (new_parts.size()) {
		particle p = new_parts.back();
		new_parts.pop_back();
		part_int index;
		{
			std::lock_guard<spinlock_type> lock(mutex);
			if (free_indices.size()) {
				index = free_indices.back();
				free_indices.pop_back();
			} else {
				index = parts->size();
				parts->resize(parts->size() + 1);
			}
		}
		parts->set_particle(p, index);
	}

}

bool particle_server::domain_decomp_gather() {
	std::vector<hpx::future<bool>> futs;
	if (hpx_rank() == 0) {
		for (int i = 1; i < hpx_size(); i++) {
			futs.push_back(hpx::async<particle_server_domain_decomp_gather_action>(hpx_localities()[i]));
		}
	}
	bool gathered_all = parts->gather_sends(part_sends, free_indices, dbounds);
	hpx::wait_all(futs.begin(),futs.end());
	for (auto& b : futs) {
		if (!b.get()) {
			gathered_all = false;
		}
	}
	return gathered_all;
}

particle_set& particle_server::get_particle_set() {
	return *parts;
}

const domain_bounds& particle_server::get_domain_bounds() {
	return dbounds;
}

void particle_server::init() {
	std::vector<hpx::future<void>> futs;
	if (hpx_rank() == 0) {
		for (int i = 1; i < hpx_size(); i++) {
			futs.push_back(hpx::async<particle_server_init_action>(hpx_localities()[i]));
		}
	}
	const size_t start = ((size_t) hpx_rank()) * (size_t) global().opts.nparts / (size_t) hpx_size();
	const size_t stop = ((size_t) hpx_rank() + 1) * (size_t) global().opts.nparts / (size_t) hpx_size();
	const size_t size = stop - start;
	if (size > std::numeric_limits<part_int>::max()) {
		ERROR()
		;
	}
	CUDA_MALLOC(parts, 1);
	new (parts) particle_set(size);
	dbounds.create_uniform_bounds();
	hpx::wait_all(futs.begin(), futs.end());
}
