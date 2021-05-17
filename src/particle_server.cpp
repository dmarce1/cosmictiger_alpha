#include <cosmictiger/particle_server.hpp>
#include <cosmictiger/global.hpp>

particle_set* particle_server::parts = nullptr;
std::vector<part_int> particle_server::free_indices;
particle_send_type particle_server::part_sends;
domain_bounds particle_server::dbounds;
spinlock_type particle_server::mutex;

HPX_PLAIN_ACTION(particle_server::init, particle_server_init_action);
HPX_PLAIN_ACTION(particle_server::domain_decomp_gather, particle_server_domain_decomp_gather_action);
HPX_PLAIN_ACTION(particle_server::domain_decomp_send, particle_server_domain_decomp_send_action);
HPX_PLAIN_ACTION(particle_server::domain_decomp_finish, particle_server_domain_decomp_finish_action);
HPX_PLAIN_ACTION(particle_server::domain_decomp_transmit, particle_server_domain_decomp_transmit_action);
HPX_PLAIN_ACTION(particle_server::generate_random, particle_server_generate_random_action);

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

void particle_server::domain_decomp_gather() {
	std::vector<hpx::future<void>> futs;
	if (hpx_rank() == 0) {
		for (int i = 1; i < hpx_size(); i++) {
			futs.push_back(hpx::async<particle_server_domain_decomp_gather_action>(hpx_localities()[i]));
		}
	}
	parts->gather_sends(part_sends, free_indices, dbounds);
	hpx::wait_all(futs.begin(), futs.end());
}

particle_set& particle_server::get_particle_set() {
	return *parts;
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
