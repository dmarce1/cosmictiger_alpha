#include <cosmictiger/particle.hpp>
#include <cosmictiger/memory.hpp>
#include <cosmictiger/rand.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/global.hpp>

#include <unordered_map>
#include <algorithm>

void particle_set::prepare_sort() {
#ifdef USE_READMOSTLY
	for (int dim = 0; dim < NDIM; dim++) {
		CUDA_CHECK(cudaMemAdvise(xptr_[dim], sizeof(fixed32) * size(), cudaMemAdviseUnsetReadMostly, 0));
	}
	CUDA_CHECK(cudaMemAdvise(uptr_, sizeof(vel_type) * size(), cudaMemAdviseUnsetReadMostly, 0));
#endif
}

void particle_set::prepare_kick() {
#ifdef USE_READMOSTLY
	for (int dim = 0; dim < NDIM; dim++) {
		CUDA_CHECK(cudaMemAdvise(xptr_[dim], sizeof(fixed32) * size(), cudaMemAdviseSetReadMostly, 0));
	}
	CUDA_CHECK(cudaMemAdvise(uptr_, sizeof(vel_type) * size(), cudaMemAdviseUnsetReadMostly, 0));
#endif
}

void particle_set::prepare_drift() {
#ifdef USE_READMOSTLY
	for (int dim = 0; dim < NDIM; dim++) {
		CUDA_CHECK(cudaMemAdvise(xptr_[dim], sizeof(fixed32) * size(), cudaMemAdviseUnsetReadMostly, 0));
	}
	CUDA_CHECK(cudaMemAdvise(uptr_, sizeof(vel_type) * size(), cudaMemAdviseSetReadMostly, 0));
#endif
}

particle_set::particle_set(size_t size, size_t offset) {
	offset_ = offset;
	size_ = size;
	virtual_ = false;
	size_t chunk_size = NDIM * sizeof(fixed32) + sizeof(vel_type);
#ifdef TEST_FORCE
	chunk_size += (NDIM + 1) * sizeof(float);
#endif
	uint8_t *data;
	//unified_allocator alloc;
	//data = (uint8_t*) alloc.allocate(chunk_size * size);
	CUDA_MALLOC(data, chunk_size * size);
	CHECK_POINTER(data);
	base_ = (void*) data;
	for (int dim = 0; dim < NDIM; dim++) {
		xptr_[dim] = (fixed32*) (data + dim * sizeof(fixed32) * size);
	}
	uptr_ = (vel_type*) (data + size * NDIM * sizeof(fixed32));
#ifdef TEST_FORCE
	const auto offset1 = sizeof(vel_type) * size + NDIM * size * sizeof(fixed32);
	for (size_t dim = 0; dim < NDIM; dim++) {
		gptr_[dim] = (float*) (data + offset1 + dim * size * sizeof(float));
	}
	eptr_ = (float*) (data + offset1 + size * NDIM * sizeof(float));
#endif
	for (int i = 0; i < size; i++) {
		uptr_[i].p.r = 0;
	}
}

void particle_set::load_from_file(FILE* fp) {
	fread(&global().opts.z0, sizeof(global().opts.z0), 1, fp);
	fread(&global().opts.omega_m, sizeof(global().opts.omega_m), 1, fp);
	fread(&global().opts.hubble, sizeof(global().opts.hubble), 1, fp);
	fread(&global().opts.code_to_cm, sizeof(global().opts.code_to_cm), 1, fp);
	fread(&global().opts.code_to_g, sizeof(global().opts.code_to_g), 1, fp);
	fread(&global().opts.code_to_s, sizeof(global().opts.code_to_s), 1, fp);
	fread(&global().opts.H0, sizeof(global().opts.H0), 1, fp);
	fread(&global().opts.G, sizeof(global().opts.G), 1, fp);
	fread(&global().opts.M, sizeof(global().opts.M), 1, fp);
	for (int dim = 0; dim < NDIM; dim++) {
		fread(xptr_[dim], sizeof(fixed32), size(), fp);
	}
	fread(uptr_, sizeof(vel_type), size(), fp);
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
	fwrite(uptr_, sizeof(vel_type), size(), fp);
}

//
//void particle_set::prefetch(size_t b, size_t e, cudaStream_t stream) {
//   for (int dim = 0; dim < NDIM; dim++) {
//      CUDA_CHECK(cudaMemPrefetchAsync((void* ) (xptr_[dim] + b), sizeof(fixed32) * (e - b), 0, stream));
//      CUDA_CHECK(cudaMemPrefetchAsync((void* ) (vptr_[dim] + b), sizeof(float) * (e - b), 0, stream));
//   }
//   CUDA_CHECK(cudaMemPrefetchAsync((void* ) (rptr_ + b), sizeof(int8_t) * (e - b), 0, stream));
//}

particle_set::~particle_set() {
	if (!virtual_) {
//		unified_allocator alloc;
//		alloc.deallocate(base_);
		CUDA_FREE(base_);
	}
}

void particle_set::generate_random() {
	for (int i = 0; i < size_; i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			pos(0, i) = rand_fixed32();
			pos(1, i) = rand_fixed32();
			pos(2, i) = rand_fixed32();
			vel(i).p.x = 0.f;
			vel(i).p.y = 0.f;
			vel(i).p.z = 0.f;
		}
		set_rung(0, i);
	}
}

void particle_set::generate_grid() {
	const auto dim = global().opts.parts_dim;
	for (size_t i = 0; i < dim; i++) {
		for (size_t j = 0; j < dim; j++) {
			for (size_t k = 0; k < dim; k++) {
				const size_t iii = i * dim * dim + j * dim + k;
				pos(0, iii) = (i + 0.5) / dim;
				pos(1, iii) = (j + 0.5) / dim;
				pos(2, iii) = (k + 0.5) / dim;
				vel(i).p.x = 0.f;
				vel(i).p.y = 0.f;
				vel(i).p.z = 0.f;
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
	fread(&dummy, sizeof(dummy), 1, fp);
	fread(header, sizeof(*header), 1, fp);
	fread(&dummy, sizeof(dummy), 1, fp);
	printf("Reading %lli particles\n", header->npart[1]);
	printf("Z =             %e\n", header->redshift);
	printf("particle mass = %e\n", header->mass[1]);
	printf("Omega_m =       %e\n", header->Omega0);
	printf("Omega_lambda =  %e\n", header->OmegaLambda);
	printf("Hubble Param =  %e\n", header->HubbleParam);
	fread(&dummy, sizeof(dummy), 1, fp);
	fclose(fp);

}

size_t particle_set::sort_range(size_t begin, size_t end, double xm, int xdim) {

	size_t lo = begin;
	size_t hi = end;
	fixed32 xmid(xm);
	while (lo < hi) {
		if (xptr_[xdim][lo] >= xmid) {
			while (lo != hi) {
				hi--;
				if (xptr_[xdim][hi] < xmid) {
					std::swap(xptr_[0][hi], xptr_[0][lo]);
					std::swap(xptr_[1][hi], xptr_[1][lo]);
					std::swap(xptr_[2][hi], xptr_[2][lo]);
					std::swap(uptr_[hi], uptr_[lo]);
//					std::swap(rptr_[hi], rptr_[lo]);
					break;
				}
			}
		}
		lo++;
	}
	return hi;
}

void particle_set::load_particles(std::string filename) {
	int4byte dummy;
	FILE *fp = fopen(filename.c_str(), "rb");
	if (!fp) {
		printf("Unable to load %s\n", filename.c_str());
		abort();
	}
	io_header_1 header;
	fread(&dummy, sizeof(dummy), 1, fp);
	fread(&header, sizeof(header), 1, fp);
	fread(&dummy, sizeof(dummy), 1, fp);
	printf("Reading %lli particles\n", header.npart[1]);
	printf("Z =             %e\n", header.redshift);
	printf("particle mass = %e\n", header.mass[1]);
	printf("Omega_m =       %e\n", header.Omega0);
	printf("Omega_lambda =  %e\n", header.OmegaLambda);
	printf("Hubble Param =  %e\n", header.HubbleParam);
	global().opts.z0 = header.redshift;
	global().opts.omega_m = header.Omega0;
	global().opts.hubble = header.HubbleParam;
	const auto Gcgs = 6.67259e-8;
	const auto ccgs = 2.99792458e+10;
	const auto Hcgs = 3.2407789e-18;
	global().opts.code_to_cm = (header.mass[1] * global().opts.nparts * 8.0 * M_PI * Gcgs * global().opts.code_to_g);
	global().opts.code_to_cm /= 3.0 * global().opts.omega_m * Hcgs * Hcgs;
	global().opts.code_to_cm = std::pow(global().opts.code_to_cm, 1.0 / 3.0);
	global().opts.code_to_s = global().opts.code_to_cm / global().opts.code_to_cms;
	global().opts.H0 = Hcgs * global().opts.code_to_s;
	global().opts.G = Gcgs / pow(global().opts.code_to_cm, 3) * global().opts.code_to_g
			* pow(global().opts.code_to_s, 2);
	double m_tot = global().opts.omega_m * 3.0 * global().opts.H0 * global().opts.H0 / (8 * M_PI * global().opts.G);
	global().opts.M = m_tot / global().opts.nparts;

	printf("G in code units = %e\n", global().opts.G);
	printf("M in code units = %e\n", global().opts.M);

	fread(&dummy, sizeof(dummy), 1, fp);
// printf( "%li\n", parts.size());
	for (int i = 0; i < header.npart[1]; i++) {
		float x, y, z;
		fread(&x, sizeof(float), 1, fp);
		fread(&y, sizeof(float), 1, fp);
		fread(&z, sizeof(float), 1, fp);
		double sep = 0.5 * std::pow(header.npart[1], -1.0 / 3.0);
		x += sep;
		y += sep;
		z += sep;
		while (x > 1.0) {
			x -= 1.0;
		}
		while (y > 1.0) {
			y -= 1.0;
		}
		while (z > 1.0) {
			z -= 1.0;
		}
		pos(0, i) = x;
		pos(1, i) = y;
		pos(2, i) = z;
//    printf( "%e %e %e\n", x, y, z);
	}
	fread(&dummy, sizeof(dummy), 1, fp);
	fread(&dummy, sizeof(dummy), 1, fp);
	const auto c0 = 1.0 / (1.0 + header.redshift);
	for (int i = 0; i < header.npart[1]; i++) {
		float vx, vy, vz;
		fread(&vx, sizeof(float), 1, fp);
		fread(&vy, sizeof(float), 1, fp);
		fread(&vz, sizeof(float), 1, fp);
		vel(i).p.x = vx * std::pow(c0, 1.5);
		vel(i).p.y = vy * std::pow(c0, 1.5);
		vel(i).p.z = vz * std::pow(c0, 1.5);
		set_rung(0, i);
	}
	fread(&dummy, sizeof(dummy), 1, fp);
	fclose(fp);
}
