#include <cosmictiger/particle_server.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/tree_database.hpp>
#include <cosmictiger/hpx.hpp>
#include <malloc.h>

particle_set* particle_server::parts = nullptr;
vector<part_int> particle_server::free_indices;
particle_send_type particle_server::part_sends;
vector<particle> particle_server::part_recvs;
domain_bounds dbounds;
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
HPX_PLAIN_ACTION(particle_server::load_NGenIC, particle_server_load_NGenIC_action);

/**** header from N-GenIC*****/

typedef int int4byte;
typedef unsigned int uint4byte;
struct io_header_1 { /* From NGenIC */
	uint4byte npart[6]; /*!< npart[1] gives the number of particles in the present file, other particle types are ignored */
	double mass[6]; /*!< mass[1] gives the particle mass */
	double time; /*!< time (=cosmological scale factor) of snapshot */
	double redshift; /*!< redshift of snapshot */
	int4byte flag_sfr; /*!< flags whether star formation is used (not available in L-Gadget2) */
	int4byte flag_feedback; /*!< flags whether feedback from star formation is included */
	uint4byte npartTotal[6]; /*!< npart[1] gives the total number of particles in the run. If this number exceeds 2^32, the npartTotal[2] stores
	 the result of a division of the particle number by 2^32, while npartTotal[1] holds the remainder. */
	int4byte flag_cooling; /*!< flags whether radiative cooling is included */
	int4byte num_files; /*!< determines the number of files that are used for a snapshot */
	double BoxSize; /*!< Simulation box size (in code units) */
	double Omega0; /*!< matter density */
	double OmegaLambda; /*!< vacuum energy density */
	double HubbleParam; /*!< little 'h' */
	int4byte flag_stellarage; /*!< flags whether the age of newly formed stars is recorded and saved */
	int4byte flag_metals; /*!< flags whether metal enrichment is included */
	int4byte hashtabsize; /*!< gives the size of the hashtable belonging to this snapshot file */
	char fill[84]; /*!< fills to 256 Bytes */
};

void particle_server::load_NGenIC() {
	int4byte dummy;
	std::string filename;
	if (hpx_size() == 1) {
		filename = "ics";
	} else {
		filename = "ics." + std::to_string(hpx_rank());
	}
	FILE *fp = fopen(filename.c_str(), "rb");
	if (!fp) {
		printf("Unable to load %s\n", filename.c_str());
		abort();
	}
	io_header_1 header;
	FREAD(&dummy, sizeof(dummy), 1, fp);
	FREAD(&header, sizeof(header), 1, fp);
	FREAD(&dummy, sizeof(dummy), 1, fp);
	size_t total_parts = size_t(header.npartTotal[1]) + (size_t(1) << size_t(32)) * size_t(header.npartTotal[2]);
	if (hpx_rank() == 0) {
		printf("Reading %li particles\n", total_parts);
		printf("Z =             %e\n", header.redshift);
		printf("Omega_m =       %e\n", header.Omega0);
		printf("Omega_lambda =  %e\n", header.OmegaLambda);
		printf("Hubble Param =  %e\n", header.HubbleParam);
	}
	options opts = global().opts;
	opts.nparts = total_parts;
	opts.parts_dim = std::round(std::pow(total_parts, -1.0 / 3.0));
	opts.z0 = header.redshift;
	opts.omega_m = header.Omega0;
	opts.hubble = header.HubbleParam;
	const auto Gcgs = 6.67259e-8;
	const auto ccgs = 2.99792458e+10;
	const auto Hcgs = 3.2407789e-18;
	opts.code_to_g = 1.99e33;
	opts.code_to_cm = (header.mass[1] * opts.nparts * 8.0 * M_PI * Gcgs * opts.code_to_g);
	opts.code_to_cm /= 3.0 * opts.omega_m * Hcgs * Hcgs;
	opts.code_to_cm = std::pow(opts.code_to_cm, 1.0 / 3.0);

	opts.code_to_cm /= header.HubbleParam;
	opts.code_to_g /= header.HubbleParam;
	opts.code_to_s = opts.code_to_cm / opts.code_to_cms;

	opts.H0 = Hcgs * opts.code_to_s;
	opts.G = Gcgs / pow(opts.code_to_cm, 3) * opts.code_to_g * pow(opts.code_to_s, 2);
	double m_tot = opts.omega_m * 3.0 * sqr(opts.H0 * opts.hubble) / (8 * M_PI * opts.G);
	opts.M = m_tot / opts.nparts;

	printf("G in code units = %e\n", opts.G);
	printf("M in code units = %e\n", opts.M);

	if (hpx_rank() == 0) {
		global_set_options(opts);
	}
	std::vector<hpx::future<void>> futs;
	if (hpx_rank() == 0) {
		for (int i = 1; i < hpx_size(); i++) {
			futs.push_back(hpx::async<particle_server_load_NGenIC_action>(hpx_localities()[i]));
		}
	}
	parts->resize(header.npart[1]);
	FREAD(&dummy, sizeof(dummy), 1, fp);
	for (int i = 0; i < header.npart[1]; i++) {
		float x, y, z;
		FREAD(&x, sizeof(float), 1, fp);
		FREAD(&y, sizeof(float), 1, fp);
		FREAD(&z, sizeof(float), 1, fp);
		double sep = 0.5 * std::pow(header.npart[1], -1.0 / 3.0);
		//x += sep;
		//y += sep;
		//z += sep;
		while (x > 1.0) {
			x -= 1.0;
		}
		while (y > 1.0) {
			y -= 1.0;
		}
		while (z > 1.0) {
			z -= 1.0;
		}
		parts->pos(0, i) = x;
		parts->pos(1, i) = y;
		parts->pos(2, i) = z;
	}
	FREAD(&dummy, sizeof(dummy), 1, fp);
	FREAD(&dummy, sizeof(dummy), 1, fp);
	const auto c0 = 1.0 / (1.0 + header.redshift);
	for (int i = 0; i < header.npart[1]; i++) {
		float vx, vy, vz;
		FREAD(&vx, sizeof(float), 1, fp);
		FREAD(&vy, sizeof(float), 1, fp);
		FREAD(&vz, sizeof(float), 1, fp);
		parts->vel(0, i) = vx * std::pow(c0, 1.5);
		parts->vel(1, i) = vy * std::pow(c0, 1.5);
		parts->vel(2, i) = vz * std::pow(c0, 1.5);
		parts->set_rung(0, i);
	}
	FREAD(&dummy, sizeof(dummy), 1, fp);
	fclose(fp);
	hpx::wait_all(futs.begin(), futs.end());
}

std::vector<fixed32> particle_server::gather_pos(std::vector<part_iters> iters) {
	std::vector<fixed32> data;
	part_int size = 0;
	//const int nthreads = 1;
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

//	PRINT("importing %i particles or %e of local in %i sets with %i requests\n", size,
//			size / (double) parts->pos_size(), remotes_sorted.size(), requests.size());
	tree_data_map_global_to_local1();
	std::unique_lock<shared_mutex_type> lock(shared_mutex);
	parts->resize_pos(parts->pos_size() + size);
	lock.unlock();
	int j = 0;
	for (auto i = requests.begin(); i != requests.end(); i++) {
		futs2.push_back(futs1[j++].then([i,&offsets](hpx::future<std::vector<fixed32>>&& fut) {
			std::vector<hpx::future<void>> futs;
			const auto data = fut.get();
			//const int nthreads = 1;
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
//#ifndef NDEBUG
											local_tree.set_proc_range(hpx_rank(), hpx_rank());
//#endif
										}
									}));
				}
				hpx::wait_all(futs.begin(),futs.end());

			}));
	}
	hpx::wait_all(futs2.begin(), futs2.end());
//	PRINT("Done filling locals\n");
}

void particle_server::check_domain_bounds() {
	std::vector<hpx::future<void>> futs;
	if (hpx_rank() == 0) {
		for (int i = 1; i < hpx_size(); i++) {
			futs.push_back(hpx::async<particle_server_check_domain_bounds_action>(hpx_localities()[i]));
		}
	}
	auto myrange = dbounds.find_proc_range(hpx_rank());
	bool killme = false;
	for (part_int i = 0; i < parts->size(); i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			const auto x = parts->pos(dim, i).to_double();
			if (x < myrange.begin[dim] || x > myrange.end[dim]) {
				PRINT("ERROR : %e %e %e\n", x, myrange.begin[dim], myrange.end[dim]);
				killme = true;
			}
		}
	}
	if (killme) {
		ABORT();
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

	parts->generate_random(1234 * hpx_rank() + 42);

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
	if (hpx_size() > 1) {
		bool complete;
		do {
			complete = domain_decomp_gather();
			domain_decomp_send();
			domain_decomp_finish();
		} while (!complete);
	}
//	check_domain_bounds();
	unified_allocator alloc;
	alloc.reset();
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

void particle_server::domain_decomp_finish() {
	std::vector<hpx::future<void>> futs;
	if (hpx_rank() == 0) {
		for (int i = 1; i < hpx_size(); i++) {
			futs.push_back(hpx::async<particle_server_domain_decomp_finish_action>(hpx_localities()[i]));
		}
	}

	parts->sort_parts(part_recvs.begin(), part_recvs.end());
//	printf("%i %i %i\n", hpx_rank(), free_indices.size(), part_recvs.size());
//	printf("rank %i %i\n", hpx_rank(), parts->size());
	int free_offset = 0;
	if (free_indices.size() < part_recvs.size()) {
		part_int start = free_indices.size();
		part_int stop = part_recvs.size();
		free_indices.resize(part_recvs.size());
		part_int j = parts->size();
		for (int i = start; i < stop; i++) {
			free_indices[i] = j++;
		}
		parts->resize(j);
	}
//	printf("rank %i %i\n", hpx_rank(), parts->size());
	int nthreads = hardware_concurrency();
	free_offset = part_recvs.size();
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([proc,nthreads]() {
			const part_int start = proc * part_recvs.size() / nthreads;
			const part_int stop = (proc+1) * part_recvs.size() / nthreads;
			for( part_int i = start; i < stop; i++) {
				assert(hpx_rank()==dbounds.find_proc(part_recvs[i].x));
	//			printf( "%e %e %e\n", part_recvs[i].x[0].to_float(), part_recvs[i].x[1].to_float(), part_recvs[i].x[2].to_float());
				parts->set_particle(part_recvs[i], free_indices[i]);
			}
		}));
	}
	hpx::wait_all(futs.begin(), futs.end());
	part_recvs = decltype(part_recvs)();
	parts->free_particles(free_indices.data() + free_offset, free_indices.size() - free_offset);
	free_indices = decltype(free_indices)();
	part_sends = decltype(part_sends)();
//	malloc_stats();
}

void particle_server::domain_decomp_transmit(vector<particle> new_parts) {
	part_int start;
	part_int stop;
	std::unique_lock<shared_mutex_type> lock(shared_mutex);
	part_int i = part_recvs.size();
	part_recvs.resize(start + new_parts.size());
	lock.unlock();
	shared_mutex.lock_shared();
	while (new_parts.size()) {
		particle p = new_parts.back();
		new_parts.pop_back();
		part_recvs[i++] = p;
//		PRINT( "%e %e %e\n", p.x[0].to_float(),  p.x[1].to_float(),  p.x[2].to_float());
	}
	shared_mutex.unlock_shared();
/*	PRINT( "-------------\n");
	for( int i = 0; i < part_recvs.size(); i++) {
		auto p = part_recvs[i];
		PRINT( "%e %e %e\n", p.x[0].to_float(),  p.x[1].to_float(),  p.x[2].to_float());
	}*/


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
