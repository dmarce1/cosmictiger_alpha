#include <cosmictiger/particle_server.hpp>
#include <cosmictiger/hpx.hpp>

#include <unordered_map>

HPX_PLAIN_ACTION(particle_server::init, particle_server_init_action);
HPX_PLAIN_ACTION(particle_server::generate_random, particle_server_generate_random_action);
HPX_PLAIN_ACTION(particle_server::local_sort, particle_server_local_sort_action);
HPX_PLAIN_ACTION(particle_server::swap_particles, particle_server_swap_particles_action);
HPX_PLAIN_ACTION(particle_server::execute_swaps, particle_server_execute_swaps_action);
HPX_PLAIN_ACTION(particle_server::read_pos_cache_line, particle_server_read_pos_cache_line_action);
HPX_PLAIN_ACTION(particle_server::read_rungs, particle_server_read_rungs_action);
HPX_PLAIN_ACTION(particle_server::write_rungs_inc_vel, particle_server_write_rungs_inc_vel_action);

struct sort_iters {
	part_iters lo;
	part_iters hi;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & lo;
		arc & hi;
	}
};

template<class Arc>
void particle_arc::save(Arc&& a, unsigned) const {
	size_t size = range_from.second - range_from.first;
	a & range_to;
	a & range_from;
	a & zero_load;
	a & size;
	if (zero_save) {
		particle_server pserv;
		auto* parts = pserv.get_particle_sets().sets[pi];
		const auto b = range_from.first;
		const auto e = range_from.second;
		for (int dim = 0; dim < NDIM; dim++) {
			for (auto i = b; i < e; i++) {
				a & parts->pos(dim, i);
			}
		}
		for (auto i = b; i < e; i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				a & parts->vel(dim, i);
			}
			a & parts->rung(i);
			if (global().opts.groups) {
				a & parts->group(i);
			}
		}
	} else {
		for( size_t i = 0; i < NDIM * size; i++) {
			a & fixed32_data[i];
		}
		for( size_t i = 0; i < NDIM * size; i++) {
			a & float_data[i];
		}
		for( size_t i = 0; i < size; i++) {
			a & int8_data[i];
		}
		if( global().opts.groups ) {
			for( size_t i = 0; i < size; i++) {
				a & ulli_data[i];
			}
		}
	}
}
template<class Arc>
void particle_arc::load(Arc&& a, unsigned) {
	size_t size;
	a & range_to;
	a & range_from;
	a & zero_load;
	a & size;
	fixed32_data.resize(size*NDIM);
	float_data.resize(size*NDIM);
	int8_data.resize(size);
	for( size_t i = 0; i < NDIM * size; i++) {
		a & fixed32_data[i];
	}
	for( size_t i = 0; i < NDIM * size; i++) {
		a & float_data[i];
	}
	for( size_t i = 0; i < size; i++) {
		a & int8_data[i];
	}
	if( global().opts.groups ) {
		ulli_data.resize(size);
		for( size_t i = 0; i < size; i++) {
			a & ulli_data[i];
		}
	}
	if (zero_load) {
		particle_server pserv;
		const auto b = range_to.first;
		const auto e = range_to.second;
		auto* parts = pserv.get_particle_sets().sets[pi];
		int j;
		j = 0;
		for (int dim = 0; dim < NDIM; dim++) {
			for (auto i = b; i < e; i++) {
				parts->pos(dim, i) = fixed32_data[j];
				j++;
			}
		}
		j = 0;
		for (auto i = b; i < e; i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				parts->vel(dim, i) = float_data[j];
				j++;
			}
		}
		j = 0;
		for (auto i = b; i < e; i++) {
			parts->set_rung(int8_data[j], i);
			j++;
		}
		if (global().opts.groups) {
			j = 0;
			for (auto i = b; i < e; i++) {
				parts->group(i) = ulli_data[j];
				j++;
			}
		}
	}
}

#define POS_CACHE_LINE_SIZE_MAX 1024

std::array<std::array<pos_cache_type, POS_CACHE_SIZE>, NPART_TYPES> particle_server::pos_caches;
std::array<std::array<mutex_type, POS_CACHE_SIZE>, NPART_TYPES> particle_server::mutexes;
particle_sets* particle_server::parts;
size_t particle_server::my_start;
size_t particle_server::my_stop;
size_t particle_server::my_size;
size_t particle_server::global_size;
size_t particle_server::pos_cache_line_size;
int particle_server::nprocs;
int particle_server::rank;
std::vector<hpx::id_type> particle_server::localities;

size_t particle_server::pos_cache_line_index(size_t i) {
	return i - (i % pos_cache_line_size);
}

pos_line_type particle_server::read_pos_cache_line(int pi, size_t i) {
	i = pos_cache_line_index(i);
	pos_line_type x;
	for (int dim = 0; dim < NDIM; dim++) {
		for (size_t j = i; j < std::min(i + pos_cache_line_size, my_stop); j++) {
			x[dim].push_back(parts->sets[pi]->pos(dim, j));
		}
	}
	return std::move(x);
}

fixed32 particle_server::pos_cache_read(int pi, int dim, size_t i) {
	static const auto localities = hpx_localities();
	size_t i0 = pos_cache_line_index(i);
	size_t cache = part_hash_lo()(i0);
	std::unique_lock<mutex_type> lock(mutexes[pi][cache]);
	auto entry_iter = pos_caches[pi][cache].find(i0);
	if (entry_iter == pos_caches[pi][cache].end()) {
		auto prms = std::make_shared<hpx::lcos::local::promise<void>>();
		auto& entry = pos_caches[pi][cache][i0];
		entry.fut = std::make_shared<hpx::shared_future<void>>(prms->get_future());
		lock.unlock();
		hpx::async([i0,cache,pi,prms]() {
			particle_server_read_pos_cache_line_action action;
			auto data = action(localities[index_to_rank(i0)], pi, i0);
			std::unique_lock<mutex_type> lock(mutexes[pi][cache]);
			auto& entry = pos_caches[pi][cache][i0];
			entry.data = std::make_shared<pos_line_type>(std::move(data));
			lock.unlock();
			prms->set_value();
		});
		lock.lock();
		entry_iter = pos_caches[pi][cache].find(i0);
	}
	auto& fut = entry_iter->second.fut;
	const auto& data = entry_iter->second.data;
	lock.unlock();
	fut->get();
	return (*data)[dim][i - i0];
}

void particle_server::init() {
	printf("Initializing particle server on rank %i\n", hpx_rank());
	std::vector<hpx::future<void>> futs;
	rank = hpx_rank();
	nprocs = hpx_size();
	localities = hpx_localities();
	if (rank == 0) {
		for (int this_rank = 1; this_rank < nprocs; this_rank++) {
			futs.push_back(hpx::async<particle_server_init_action>(localities[this_rank]));
		}
	}
	global_size = global().opts.nparts;
	my_start = (rank * global_size) / nprocs;
	my_stop = ((rank + 1) * global_size) / nprocs;
	printf("range on rank %i is %li %li %li\n", rank, my_start, my_stop, global_size);
	my_size = my_stop - my_start;
	parts = new particle_sets(my_size, my_start);
	pos_cache_line_size = std::min((size_t) POS_CACHE_LINE_SIZE_MAX, global_size / nprocs);
	printf("Done on rank %i\n", rank);
	hpx::wait_all(futs.begin(), futs.end());
}

void particle_server::generate_random() {
	std::vector<hpx::future<void>> futs;
	if (rank == 0) {
		for (int this_rank = 1; this_rank < nprocs; this_rank++) {
			futs.push_back(hpx::async<particle_server_generate_random_action>(localities[this_rank]));
		}
	}
	const int npart_types = global().opts.sph ? 2 : 1;
	for (int pi = 0; pi < npart_types; pi++) {
		parts->sets[pi]->generate_random(rank * 1234 + 42 + 333 * pi);
	}
	hpx::wait_all(futs.begin(), futs.end());
}

int particle_server::index_to_rank(size_t index) {
	return std::min((int) ((index + int(global_size % nprocs != 0)) * nprocs / global_size), nprocs - 1);
}

size_t particle_server::local_sort(int pi, size_t begin, size_t end, double xmid, int xdim) {
	printf("local sort begin : %i %li %li\n", rank, begin, end);
	const size_t part_mid = parts->sets[pi]->sort_range(begin, end, xmid, xdim);
	printf("local sort done  : %i %li %li %li\n", rank, begin, part_mid, end);
	return part_mid;
}

particle_arc particle_server::swap_particles(int pi, particle_arc arc) {
	parts->sets[pi]->swap_particle_archive(arc);
	arc.zero_save = false;
	arc.zero_load = true;
	arc.pi = pi;
	swap(arc.range_to, arc.range_from);
	return arc;
}

void particle_server::execute_swaps(int pi, std::vector<sort_quantum> swaps) {
	std::vector<hpx::future<particle_arc>> futs;
	for (const auto swap : swaps) {
		printf("Executing swap %li - %li / %li - %li between %i and %i\n", swap.range_from.first, swap.range_from.second,
				swap.range_to.first, swap.range_to.second, swap.rank_from, swap.rank_to);
//		auto parc = parts->sets[pi]->save_particle_archive(swap.range_from.first, swap.range_from.second);
		particle_arc parc;
		parc.zero_save = true;
		parc.zero_load = false;
		parc.pi = pi;
		parc.range_to = swap.range_to;
		parc.range_from = swap.range_from;
		futs.push_back(hpx::async<particle_server_swap_particles_action>(localities[swap.rank_to], pi, std::move(parc)));
	}
	for (auto& f : futs) {
		auto arc = f.get();
//		std::swap(arc.range_from.first, arc.range_to.first);
//		std::swap(arc.range_from.second, arc.range_to.second);
//		parts->sets[pi]->load_particle_archive(std::move(arc));
	}
}

int particle_server::compute_target_rank(parts_type pranges) {
	particle_server pserv;
	const int npart_types = global().opts.sph ? 2 : 1;
	bool all_same = true;
	int target_rank = pserv.index_to_rank(pranges[0].first);
	for (int pi = 0; pi < npart_types; pi++) {
		if (pserv.index_to_rank(pranges[pi].first) != target_rank) {
			all_same = false;
			break;
		}
		if (pserv.index_to_rank(pranges[pi].second - 1) != target_rank) {
			all_same = false;
			break;
		}
	}
	if (!all_same) {
		for (int pi = 0; pi < npart_types; pi++) {
			std::unordered_map<int, size_t> counts;
			int rank_start = pserv.index_to_rank(pranges[pi].first);
			int rank_stop = pserv.index_to_rank(pranges[pi].second - 1);
			for (int this_rank = rank_start; this_rank <= rank_stop; this_rank++) {
				const size_t this_b = std::max(pranges[pi].first, this_rank * global_size / nprocs);
				const size_t this_e = std::min(pranges[pi].second, (this_rank + 1) * global_size / nprocs);
				if (counts.find(this_rank) == counts.end()) {
					counts[this_rank] = this_e - this_b;
				} else {
					counts[this_rank] += this_e - this_b;
				}
			}
			size_t highest_count = 0;
			for (auto i = counts.begin(); i != counts.end(); i++) {
				if (i->second > highest_count) {
					highest_count = i->second;
					target_rank = i->first;
				}
			}
		}
	}
	return target_rank;
}

void particle_server::write_rungs_inc_vel(int pi, part_iters range, std::vector<rung_t> rungs,
		std::vector<std::array<float, NDIM>> dv) {
	const int rank_start = index_to_rank(range.first);
	const int rank_stop = index_to_rank(range.second - 1);
	if (rank_start == rank_stop && rank_start == rank) {
		for (size_t i = range.first; i != range.second; i++) {
			parts->sets[pi]->set_rung(rungs[i - range.first], i);
			for (int dim = 0; dim < NDIM; dim++) {
				auto& v = parts->sets[pi]->vel(dim, i);
				v += dv[i - range.first][dim];
			}
		}
	} else {
		std::vector<hpx::future<void>> futs;
		for (int this_rank = rank_start; this_rank <= rank_stop; this_rank++) {
			const size_t this_b = std::max(range.first, this_rank * global_size / nprocs);
			const size_t this_e = std::min(range.second, (this_rank + 1) * global_size / nprocs);
			std::vector<rung_t> these_rungs;
			std::vector<std::array<float, NDIM>> these_dvs;
			for (size_t i = this_b; i != this_e; i++) {
				const int index = i - range.first;
				these_rungs.push_back(rungs[index]);
				these_dvs.push_back(dv[index]);
			}
			part_iters this_range;
			this_range.first = this_b;
			this_range.second = this_e;
			futs.push_back(
					hpx::async<particle_server_write_rungs_inc_vel_action>(localities[this_rank], pi, this_range,
							std::move(these_rungs), std::move(these_dvs)));
		}
		hpx::wait_all(futs.begin(), futs.end());
	}
}

std::vector<rung_t> particle_server::read_rungs(int pi, size_t b, size_t e) {
	std::vector<rung_t> rungs;
	const size_t size = e - b;
	rungs.reserve(size);
	const int rank_start = index_to_rank(b);
	const int rank_stop = index_to_rank(e - 1);

	if (rank_start == rank_stop && rank_start == rank) {
		for (size_t i = b; i < e; i++) {
			rungs.push_back(parts->sets[pi]->rung(i));
		}
	} else {
		printf("REMOTE RUNGS %li %li %i %i\n", b, e, rank_start, rank_stop);
		std::vector<hpx::future<std::vector<rung_t>>>futs;
		for (int this_rank = rank_start; this_rank <= rank_stop; this_rank++) {
			const size_t this_b = std::max(b, this_rank * global_size / nprocs);
			const size_t this_e = std::min(e, (this_rank + 1) * global_size / nprocs);
			futs.push_back(hpx::async<particle_server_read_rungs_action>(localities[this_rank], pi, this_b, this_e));
		}
		for (auto& f : futs) {
			auto these_rungs = f.get();
			for (const auto& rung : these_rungs) {
				rungs.push_back(rung);
			}
		}
	}
	assert(rungs.size() == size);
	return std::move(rungs);
}

size_t particle_server::sort(int pi, size_t begin, size_t end, double xmid, int xdim) {

	const int rank_start = index_to_rank(begin);
	const int rank_stop = index_to_rank(end - 1);
	size_t part_mid;
	if (rank_start == rank_stop) {
		//	printf("Local sort on range %li - %li around %e in dimension %i\n", begin, end, xmid, xdim);
		if (rank == rank_start) {
			part_mid = parts->sets[pi]->sort_range(begin, end, xmid, xdim);
		} else {
			part_mid = particle_server_local_sort_action()(localities[rank_start], pi, begin, end, xmid, xdim);
		}
	} else {
		printf("Global sort on range %li - %li around %e in dimension %i\n", begin, end, xmid, xdim);
		std::vector<hpx::future<size_t>> futs;
		const int num_ranks = rank_stop - rank_start + 1;
		futs.reserve(num_ranks);
		printf("Doing local sorts\n");
		for (int this_rank = rank_start; this_rank <= rank_stop; this_rank++) {
			const size_t this_begin = std::max(begin, this_rank * global_size / nprocs);
			const size_t this_end = std::min(end, (this_rank + 1) * global_size / nprocs);
			futs.push_back(
					hpx::async<particle_server_local_sort_action>(localities[this_rank], pi, this_begin, this_end, xmid,
							xdim));
		}
		std::vector<sort_iters> sort_ranges(num_ranks);
		size_t lo_cnt = 0;
		printf("Computing local ranges\n");
		for (int this_rank = rank_start; this_rank <= rank_stop; this_rank++) {
			const size_t this_begin = std::max(begin, this_rank * global_size / nprocs);
			const size_t this_end = std::min(end, (this_rank + 1) * global_size / nprocs);
			const int index = this_rank - rank_start;
			sort_ranges[index].lo.first = this_begin;
			sort_ranges[index].lo.second = sort_ranges[index].hi.first = futs[index].get();
			sort_ranges[index].hi.second = this_end;
			lo_cnt += sort_ranges[index].lo.second - sort_ranges[index].lo.first;
		}
		part_mid = begin + lo_cnt;
		const int rank_mid = index_to_rank(part_mid);
		printf("lo_cnt = %li mid part is %li mid_rank is %i\n", lo_cnt, part_mid, rank_mid);
		printf("Computing global sort strategy\n");
		auto& rng = sort_ranges[rank_mid - rank_start];
		if (rng.lo.second < part_mid) {
			rng.hi.first = rng.lo.first = rng.lo.second;
			rng.hi.second = part_mid;
		} else {
			rng.lo.second = rng.hi.second = rng.hi.first;
			rng.lo.first = part_mid;
		}
		std::vector<sort_quantum> swaps;
		int hi_rank = rank_mid;
		for (int lo_rank = rank_start; lo_rank <= rank_mid; lo_rank++) {
			auto& lorange = sort_ranges[lo_rank - rank_start];
			size_t hiinlo_count = lorange.hi.second - lorange.hi.first;
			while (hiinlo_count > 0) {
				if (hi_rank > rank_stop) {
					ERROR()
					;
				}
				auto& hirange = sort_ranges[hi_rank - rank_start];
				size_t loinhi_count = hirange.lo.second - hirange.lo.first;
				if (loinhi_count) {
					size_t count = std::min((size_t) SORT_SWAP_MAX, std::min(hiinlo_count, loinhi_count));
					sort_quantum sq;
					sq.rank_from = lo_rank;
					sq.rank_to = hi_rank;
					sq.range_from.first = lorange.hi.first;
					sq.range_from.second = lorange.hi.first + count;
					sq.range_to.first = hirange.lo.second - count;
					sq.range_to.second = hirange.lo.second;
					printf("---> %i %i %li %li %li %li\n", sq.rank_from, sq.rank_to, sq.range_from.first,
							sq.range_from.second, sq.range_to.first, sq.range_to.second);
					swaps.push_back(sq);
					lorange.hi.first += count;
					hirange.lo.second -= count;
				} else {
					hi_rank++;
				}
				hiinlo_count = lorange.hi.second - lorange.hi.first;
			}
		}
		std::vector<std::vector<sort_quantum>> swaps_per_rank(rank_mid - rank_start + 1);
		while (swaps.size()) {
			swaps_per_rank[swaps.back().rank_from - rank_start].push_back(swaps.back());
			swaps.pop_back();
		}
		printf("Executing sort strategy\n");
		std::vector<hpx::future<void>> swap_futs;
		for (int this_rank = rank_start; this_rank <= rank_mid; this_rank++) {
			swap_futs.push_back(
					hpx::async<particle_server_execute_swaps_action>(localities[this_rank], pi,
							std::move(swaps_per_rank[this_rank - rank_start])));
		}
		hpx::wait_all(swap_futs.begin(), swap_futs.end());
		printf("Done\n");
	}
	return part_mid;
}

particle_sets& particle_server::get_particle_sets() {
	return *parts;
}

