#include <cosmictiger/particle.hpp>
#include <cosmictiger/memory.hpp>
#include <cosmictiger/rand.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/math.hpp>

#include <silo.h>

#include <unordered_map>
#include <algorithm>

void particle_set::init_groups() {
	CUDA_MALLOC(lidptr1_, size());
	CUDA_MALLOC(lidptr2_, size());
	for (int i = 0; i < size(); i++) {
		set_last_group(i, group(i));
		group(i) = NO_GROUP;
	}
}

void particle_set::finish_groups() {
	CUDA_FREE(lidptr1_);
	CUDA_FREE(lidptr2_);
}

#define PART_BUFFER 1.2

particle particle_set::get_particle(part_int i) {
	particle p;
	for (int dim = 0; dim < NDIM; dim++) {
		p.x[dim] = pos(dim, i);
		p.v[dim] = vel(dim, i);
#ifdef TEST_FORCE
		p.g[dim] = force(dim, i);
#endif
	}
#ifdef TEST_FORCE
	p.phi = pot(i);
#endif
	p.rung = rung(i);
	static bool groups = global().opts.groups;
	if (groups) {
		p.group = group(i);
	}
	return p;
}

#define PARTICLE_SET_MAX_SEND (8*1024*1024)

bool particle_set::gather_sends(particle_send_type& sends, std::vector<part_int>& free_indices, domain_bounds bounds) {
	const auto my_range = bounds.find_proc_range(hpx_rank());
	static std::atomic<int> finished_cnt;
	static part_int max_send_parts = std::min(part_int(size() * 0.1), part_int(PARTICLE_SET_MAX_SEND));
	finished_cnt = 0;
	for (int dim = 0; dim < NDIM; dim++) {
		//	printf( "%i %e %e\n", hpx_rank(), my_range.begin[dim], my_range.end[dim]);
	}
	for (int i = 0; i < hpx_size(); i++) {
		sends[i].parts = std::vector<particle>();
	}

	const int num_threads = hardware_concurrency();
	static spinlock_type mutex;
	std::vector<hpx::future<void>> futs;
	for (int i = 0; i < num_threads; i++) {
		const auto func = [i,this,num_threads,my_range, bounds,&sends,&free_indices]() {
			const part_int start = size_t(i) * size_t(size_) / size_t(num_threads);
			const part_int stop = size_t(i+1) * size_t(size_) / size_t(num_threads);
			bool finished = true;
			for( part_int j = start; j < stop; j++) {
				bool foreign = false;
				for( int dim = 0; dim < NDIM; dim++) {
					const auto x = pos(dim,j).to_double();
					if(x < my_range.begin[dim] || x > my_range.end[dim]) {
						foreign = true;
						break;
					}
				}
				if( foreign ) {
					array<fixed32,NDIM> x;
					for( int dim = 0; dim < NDIM; dim++) {
						x[dim] = pos(dim,j);
					}
					const int proc = bounds.find_proc(x);
					particle p = get_particle(j);
					std::lock_guard<spinlock_type> lock(mutex);
					auto& entry = sends[proc];
					free_indices.push_back(j);
					auto& pts = entry.parts;
					//			printf( "%li\n", pts.size());
				pts.push_back(p);
				if( free_indices.size() >= max_send_parts ) {
					finished = false;
					break;
				}
			}
		}
		if( finished ) {
			finished_cnt++;
		}
	}	;
		futs.push_back(hpx::async(func));
	}
	hpx::wait_all(futs.begin(), futs.end());
	return (int) finished_cnt == num_threads;
}

void particle_set::free_particles(std::vector<part_int>& free_indices) {
	std::sort(free_indices.begin(), free_indices.end(), [](part_int a, part_int b) {
		return a > b;
	});
	for (auto i : free_indices) {
		const particle p = get_particle(size() - 1);
		set_particle(p, i);
		resize(size() - 1);
	}
}

void particle_set::set_particle(const particle& p, part_int i) {
	for (int dim = 0; dim < NDIM; dim++) {
		xptr_[dim][i] = p.x[dim];
		uptr_[i][dim] = p.v[dim];
#ifdef TEST_FORCE
		gptr_[dim][i] = p.g[dim];
#endif
	}
#ifdef TEST_FORCE
	eptr_[i] = p.phi;
#endif
	rptr_[i] = p.rung;
	static bool groups = global().opts.groups;
	if (groups) {
		idptr_[i] = p.group;
	}
}

particle_set::particle_set() {
	size_ = 0;
	pos_size_ = 0;
	cap_ = 0;
	pos_cap_ = 0;
#ifdef TEST_FORCE
	for (int dim = 0; dim < NDIM; dim++) {
		gptr_[dim] = nullptr;
	}
	eptr_ = nullptr;
#endif
	for (int dim = 0; dim < NDIM; dim++) {
		xptr_[dim] = nullptr;
	}
	uptr_ = nullptr;
	idptr_ = nullptr;
	rptr_ = nullptr;
	lidptr1_ = nullptr;
	lidptr2_ = nullptr;
}

template<class T>
void realloc(T*& old, part_int oldsz, part_int new_cap) {
	assert(oldsz <= new_cap);
	T* new_;
	CUDA_MALLOC(new_, new_cap);
	if (oldsz) {
		memcpy(new_, old, sizeof(T) * oldsz);
		CUDA_FREE(old);
	}
	old = new_;
}

void particle_set::resize(part_int sz) {
	if (cap_ < sz) {
		cap_ = 1024;
		while (cap_ < sz) {
			cap_ = PART_BUFFER * cap_;
		}
		resize_pos(std::max(sz, pos_size_));
		realloc(uptr_, size_, cap_);
		realloc(rptr_, size_, cap_);
#ifdef TEST_FORCE
		for (int dim = 0; dim < NDIM; dim++) {
			realloc(gptr_[dim], size_, cap_);
		}
		realloc(eptr_, size_, cap_);
#endif
		if (global().opts.groups) {
			realloc(idptr_, size_, cap_);
		}
	}
	size_ = sz;
}
void particle_set::resize_pos(part_int sz) {
	if (pos_cap_ < sz) {
		pos_cap_ = 1024;
		while (pos_cap_ < sz) {
			pos_cap_ = PART_BUFFER * pos_cap_;
		}
		for (int dim = 0; dim < NDIM; dim++) {
			realloc(xptr_[dim], pos_size_, pos_cap_);
		}
	}
	pos_size_ = sz;
}

particle_set::particle_set(part_int size) {
	size_ = 0;
	cap_ = 0;
	if (size) {
		resize(size);
	}
}

void particle_set::load_from_file(FILE* fp) {
	options opts = global().opts;
	const auto z0 = opts.z0;
	FREAD(&opts.z0, sizeof(opts.z0), 1, fp);
	FREAD(&opts.omega_m, sizeof(opts.omega_m), 1, fp);
	FREAD(&opts.hubble, sizeof(opts.hubble), 1, fp);
	FREAD(&opts.code_to_cm, sizeof(opts.code_to_cm), 1, fp);
	FREAD(&opts.code_to_g, sizeof(opts.code_to_g), 1, fp);
	FREAD(&opts.code_to_s, sizeof(opts.code_to_s), 1, fp);
	FREAD(&opts.H0, sizeof(opts.H0), 1, fp);
	FREAD(&opts.G, sizeof(opts.G), 1, fp);
	FREAD(&opts.M, sizeof(opts.M), 1, fp);
	double m_tot = opts.omega_m * 3.0 * sqr(opts.H0 * opts.hubble) / (8 * M_PI * std::abs(opts.G));
	opts.M = m_tot / opts.nparts;
	if (opts.glass_file != "") {
		opts.z0 = z0;
		opts.G = std::abs(opts.G);
	}
	if (hpx_rank() == 0) {
		global_set_options(opts);
	}
	for (int dim = 0; dim < NDIM; dim++) {
		printf("%c positions...", 'x' + dim);
		fflush(stdout);
		FREAD(xptr_[dim], sizeof(fixed32), size(), fp);
	}
	printf("velocities...");
	fflush(stdout);
	FREAD(uptr_, sizeof(array<float,NDIM>), size(), fp);
	printf("rungs...");
	fflush(stdout);
	FREAD(rptr_, sizeof(rung_t), size(), fp);
	printf("groups...");
	fflush(stdout);
	if (opts.groups) {
		FREAD(idptr_, sizeof(group_t), size(), fp);
	}
	printf("\n");
}

void particle_set::save_to_file(FILE* fp) {
	fwrite(&global().opts.z0, sizeof(global().opts.z0), 1, fp);
	fwrite(&global().opts.omega_m, sizeof(global().opts.omega_m), 1, fp);
	fwrite(&global().opts.hubble, sizeof(global().opts.hubble), 1, fp);
	fwrite(&global().opts.code_to_cm, sizeof(global().opts.code_to_cm), 1, fp);
	fwrite(&global().opts.code_to_g, sizeof(global().opts.code_to_g), 1, fp);
	fwrite(&global().opts.code_to_s, sizeof(global().opts.code_to_s), 1, fp);
	fwrite(&global().opts.H0, sizeof(global().opts.H0), 1, fp);
	fwrite(&global().opts.G, sizeof(global().opts.G), 1, fp);
	fwrite(&global().opts.M, sizeof(global().opts.M), 1, fp);
	for (int dim = 0; dim < NDIM; dim++) {
		fwrite(xptr_[dim], sizeof(fixed32), size(), fp);
	}
	fwrite(uptr_, sizeof(std::array<float, NDIM>), size(), fp);
	fwrite(rptr_, sizeof(rung_t), size(), fp);
	if (global().opts.groups) {
		fwrite(idptr_, sizeof(group_t), size(), fp);
	}
}

//
//void particle_set::prefetch(part_int b, part_int e, cudaStream_t stream) {
//   for (int dim = 0; dim < NDIM; dim++) {
//      CUDA_CHECK(cudaMemPrefetchAsync((void* ) (xptr_[dim] + b), sizeof(fixed32) * (e - b), 0, stream));
//      CUDA_CHECK(cudaMemPrefetchAsync((void* ) (vptr_[dim] + b), sizeof(float) * (e - b), 0, stream));
//   }
//   CUDA_CHECK(cudaMemPrefetchAsync((void* ) (rptr_ + b), sizeof(int8_t) * (e - b), 0, stream));
//}

particle_set::~particle_set() {
}

void particle_set::generate_grid() {
	const auto dim = global().opts.parts_dim;
	for (part_int i = 0; i < dim; i++) {
		for (part_int j = 0; j < dim; j++) {
			for (part_int k = 0; k < dim; k++) {
				const part_int iii = i * dim * dim + j * dim + k;
				pos(0, iii) = (i + 0.5) / dim;
				pos(1, iii) = (j + 0.5) / dim;
				pos(2, iii) = (k + 0.5) / dim;
				vel(0, i) = 0.f;
				vel(1, i) = 0.f;
				vel(2, i) = 0.f;
				set_rung(0, i);
			}
		}
	}
}

/**** header from N-GenIC*****/

typedef int int4byte;
typedef unsigned int uint4byte;

struct io_header_1 {
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

void load_header(io_header_1* header, std::string filename);

void load_header(io_header_1 *header, std::string filename) {
	int4byte dummy;
	FILE *fp = fopen(filename.c_str(), "rb");
	if (!fp) {
		printf("Unable to load %s\n", filename.c_str());
		abort();
	}
	FREAD(&dummy, sizeof(dummy), 1, fp);
	FREAD(header, sizeof(*header), 1, fp);
	FREAD(&dummy, sizeof(dummy), 1, fp);
	printf("Reading %i particles\n", header->npart[1]);
	printf("Z =             %e\n", header->redshift);
	printf("particle mass = %e\n", header->mass[1]);
	printf("Omega_m =       %e\n", header->Omega0);
	printf("Omega_lambda =  %e\n", header->OmegaLambda);
	printf("Hubble Param =  %e\n", header->HubbleParam);
	FREAD(&dummy, sizeof(dummy), 1, fp);
	fclose(fp);

}

part_int particle_set::sort_range(part_int begin, part_int end, double xm, int xdim) {

	part_int lo = begin;
	part_int hi = end;
	fixed32 xmid(xm);
	auto& xptr_dim = xptr_[xdim];
	auto& x = xptr_[0];
	auto& y = xptr_[1];
	auto& z = xptr_[2];
	const bool groups = global().opts.groups;
	while (lo < hi) {
		if (xptr_dim[lo] >= xmid) {
			while (lo != hi) {
				hi--;
				if (xptr_dim[hi] < xmid) {
					std::swap(x[hi], x[lo]);
					std::swap(y[hi], y[lo]);
					std::swap(z[hi], z[lo]);
					std::swap(uptr_[hi][0], uptr_[lo][0]);
					std::swap(uptr_[hi][1], uptr_[lo][1]);
					std::swap(uptr_[hi][2], uptr_[lo][2]);
					std::swap(rptr_[hi], rptr_[lo]);
					if (groups) {
						std::swap(idptr_[hi], idptr_[lo]);
					}
					break;
				}
			}
		}
		lo++;
	}
	return hi;
}

