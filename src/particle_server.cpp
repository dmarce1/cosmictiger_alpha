#include <cosmictiger/particle_server.hpp>
#include <cosmictiger/hpx.hpp>

#include <unordered_map>

HPX_PLAIN_ACTION(particle_server::init, particle_server_init_action);
HPX_PLAIN_ACTION(particle_server::generate_random, particle_server_generate_random_action);
HPX_PLAIN_ACTION(particle_server::local_sort, particle_server_local_sort_action);
HPX_PLAIN_ACTION(particle_server::swap_particles, particle_server_swap_particles_action);
HPX_PLAIN_ACTION(particle_server::execute_swaps, particle_server_execute_swaps_action);

struct sort_iters {
	part_iters lo;
	part_iters hi;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & lo;
		arc & hi;
	}
};

particle_sets* particle_server::parts;

size_t particle_server::my_start;
size_t particle_server::my_stop;
size_t particle_server::my_size;
size_t particle_server::global_size;
int particle_server::rank;
int particle_server::nprocs;
std::vector<hpx::id_type> particle_server::localities;

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
	printf( "range on rank %i is %li %li %li\n", rank,  my_start, my_stop, global_size);
	my_size = my_stop - my_start;
	parts = new particle_sets(my_size, my_start);
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

void particle_server::swap_particles(int pi, particle_arc arc) {
	parts->sets[pi]->load_particle_archive(std::move(arc));
}

void particle_server::execute_swaps(int pi, std::vector<sort_quantum> swaps) {
	std::vector<hpx::future<void>> futs;
	for (const auto swap : swaps) {
		printf("Executing swap %li - %li / %li - %li between %i and %i\n", swap.range_from.first, swap.range_from.second,
				swap.range_to.first, swap.range_to.second, swap.rank_from, swap.rank_to);
		auto parc = parts->sets[pi]->save_particle_archive(swap.range_from.first, swap.range_from.second);
		parc.range = swap.range_to;
		futs.push_back(hpx::async<particle_server_swap_particles_action>(localities[swap.rank_to], pi, std::move(parc)));
	}
	hpx::wait_all(futs.begin(), futs.end());
}

size_t particle_server::sort(int pi, size_t begin, size_t end, double xmid, int xdim) {

	const int rank_start = index_to_rank(begin);
	const int rank_stop = index_to_rank(end);
	size_t part_mid;
	if (rank_start == rank_stop) {
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
		const int rank_mid = index_to_rank(begin + part_mid);
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
					size_t count = std::min(hiinlo_count, loinhi_count);
					sort_quantum sq;
					sq.rank_from = lo_rank;
					sq.rank_to = hi_rank;
					sq.range_from.first = lorange.hi.first;
					sq.range_from.second = lorange.hi.first + count;
					sq.range_to.first = hirange.lo.second - count;
					sq.range_to.second = hirange.lo.second;
					printf("---> %i %i %li %li %li %li\n", sq.rank_from, sq.rank_to, sq.range_from.first, sq.range_from.second,
							sq.range_to.first, sq.range_to.second);
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
