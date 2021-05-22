#include <cosmictiger/particle_server.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/tree_database.hpp>
#include <cosmictiger/hpx.hpp>

particle_set* particle_server::parts = nullptr;
std::vector<part_int> particle_server::free_indices;
particle_send_type particle_server::part_sends;
domain_bounds particle_server::dbounds;
spinlock_type particle_server::mutex;
shared_mutex_type particle_server::shared_mutex;

HPX_PLAIN_ACTION(particle_server::init, particle_server_init_action);
HPX_PLAIN_ACTION(particle_server::domain_decomp_gather, particle_server_domain_decomp_gather_action);
HPX_PLAIN_ACTION(particle_server::domain_decomp_send, particle_server_domain_decomp_send_action);
HPX_PLAIN_ACTION(particle_server::domain_decomp_finish, particle_server_domain_decomp_finish_action);
HPX_PLAIN_ACTION(particle_server::domain_decomp_transmit, particle_server_domain_decomp_transmit_action);
HPX_PLAIN_ACTION(particle_server::generate_random, particle_server_generate_random_action);
HPX_PLAIN_ACTION(particle_server::check_domain_bounds, particle_server_check_domain_bounds_action);
HPX_PLAIN_ACTION(particle_server::gather_pos, particle_server_gather_pos_action);

std::vector<fixed32> particle_server::gather_pos(std::vector<part_iters> iters) {
	std::vector<fixed32> data;
	part_int size = 0;
//	const int nthreads = 1;
	const int nthreads = hardware_concurrency();
	std::vector<part_int> offsets(nthreads + 1);
	offsets[0] = 0;
	for (int proc = 0; proc < nthreads; proc++) {
		offsets[proc + 1] = offsets[proc];
		const int start = proc * iters.size() / nthreads;
		const int stop = (proc + 1) * iters.size() / nthreads;
		for (int i = start; i < stop; i++) {
			const part_int this_size = iters[i].second - iters[i].first;
			offsets[proc + 1] += this_size;
			size += this_size;
		}
	}
	data.resize(NDIM * size);
	std::vector<hpx::future<void>> futs;
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([&offsets,&data,proc,&iters,nthreads]() {
			const int start = proc * iters.size() / nthreads;
			const int stop = (proc+1) * iters.size() / nthreads;
			shared_mutex.lock_shared();
			part_int j = offsets[proc];
			for( int i = start; i < stop; i++) {
				for (part_int k = iters[i].first; k < iters[i].second; k++) {
					for (int dim = 0; dim < NDIM; dim++) {
						data[dim+NDIM*j] = parts->pos(dim, k);
					}
					j++;
				}
			}
			shared_mutex.unlock_shared();
			assert(j==offsets[proc+1]);
		}));
	}
	hpx::wait_all(futs.begin(), futs.end());
	return std::move(data);
}

void particle_server::global_to_local(std::unordered_set<tree_ptr, tree_hash> remotes_unsorted) {
	std::unordered_map<int, std::vector<tree_ptr>> requests;
	std::unordered_map<int, part_int> offsets;

	struct sort_entry {
		tree_ptr tree;
		part_int pbegin;
	};
	std::vector<sort_entry> remotes_sorted;
	remotes_sorted.reserve(remotes_unsorted.size());
	for (auto tree : remotes_unsorted) {
		sort_entry entry;
		entry.tree = tree;
		entry.pbegin = tree.get_parts().first;
		remotes_sorted.push_back(entry);
	}
	std::sort(remotes_sorted.begin(), remotes_sorted.end(), [](const sort_entry& a, const sort_entry& b) {
		return a.pbegin < b.pbegin;
	});
	for (auto entry : remotes_sorted) {
		requests[entry.tree.rank].push_back(entry.tree);
	}
	std::vector<hpx::future<std::vector<fixed32>>>futs1;
	std::vector<hpx::future<void>> futs2;
	part_int size = 0;
	for (auto i = requests.begin(); i != requests.end(); i++) {
		std::vector<part_iters> iters;
		iters.reserve(i->second.size());
		offsets[i->first] = size + parts->size();
		for (int j = 0; j < i->second.size(); j++) {
			const auto rng = i->second[j].get_parts();
			iters.push_back(rng);
			const auto dif = rng.second - rng.first;
			size += dif;
		}
		futs1.push_back(hpx::async<particle_server_gather_pos_action>(hpx_localities()[i->first], std::move(iters)));
	}

	PRINT("importing %i particles or %e of local in %i sets with %i requests\n", size,
			size / (double) parts->pos_size(), remotes_sorted.size(), requests.size());
	tree_data_map_global_to_local();
	std::unique_lock<shared_mutex_type> lock(shared_mutex);
	parts->resize_pos(parts->pos_size() + size);
	lock.unlock();
	int j = 0;
	for (auto i = requests.begin(); i != requests.end(); i++) {
		futs2.push_back(futs1[j++].then([i,&offsets](hpx::future<std::vector<fixed32>>&& fut) {
			std::vector<hpx::future<void>> futs;
			const auto data = fut.get();
//			const int nthreads = 1;
			const int nthreads = hardware_concurrency();
			std::vector<part_int> joffsets(nthreads);
			joffsets[0] = 0;
			for( int proc = 0; proc < nthreads - 1; proc++) {
				joffsets[proc + 1] = joffsets[proc];
				part_int start = proc * i->second.size() / nthreads;
				part_int stop = (proc+1) * i->second.size() / nthreads;
				for( int k = start; k < stop; k++) {
					tree_ptr local_tree = tree_data_global_to_local(i->second[k]);
					const auto rng = local_tree.get_parts();
					joffsets[proc + 1] += rng.second - rng.first;
				}
			}
			for( int proc = 0; proc < nthreads; proc++) {
				futs.push_back(hpx::async([&offsets,&joffsets,&data,proc,nthreads,i]() {
									part_int j = 0;
									part_int start = proc * i->second.size() / nthreads;
									part_int stop = (proc+1) * i->second.size() / nthreads;
									part_int offset = joffsets[proc] + offsets[i->first];
									for( int k = start; k < stop; k++) {
										assert(i->second[k].rank != hpx_rank());
										tree_ptr local_tree = tree_data_global_to_local(i->second[k]);
										const auto rng = local_tree.get_parts();
										part_iters local_iter;
										local_iter.first = j + offset;
										for( part_int l = rng.first; l < rng.second; l++) {
											assert( !(j + offset > parts->pos_size() || j + offset < 0) );
											assert( j + offset >= parts->size() );
											for( int dim = 0; dim < NDIM; dim++) {
												parts->pos(dim, j + offset) = data[dim + NDIM * (j + joffsets[proc])];
											}
											j++;
										}
										local_iter.second = j + offset;
										local_tree.set_parts(local_iter);
#ifndef NDEBUG
										local_tree.set_proc_range(hpx_rank(), hpx_rank());
#endif
									}
								}));
			}
			hpx::wait_all(futs.begin(),futs.end());

		}));
	}
	hpx::wait_all(futs2.begin(), futs2.end());
	PRINT("Done filling locals\n");
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
			assert(x >= myrange.begin[dim] && x <= myrange.end[dim]);
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

	parts->generate_random(time(NULL) + 1234*hpx_rank() + 42);

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
	hpx::wait_all(futs.begin(), futs.end());
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
	assert(size <= std::numeric_limits<part_int>::max());
	CUDA_MALLOC(parts, 1);
	new (parts) particle_set(size);
	dbounds.create_uniform_bounds();
	hpx::wait_all(futs.begin(), futs.end());
}
