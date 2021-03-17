#include <cosmictiger/particle.hpp>
#include <cosmictiger/memory.hpp>
#include <cosmictiger/rand.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/global.hpp>

#include <unordered_map>
#include <algorithm>

void particle_set::prepare_kick( cudaStream_t stream) {
	return;
	size_t begin = 0;
	for (int dim = 0; dim < NDIM; dim++) {
		CUDA_CHECK(cudaMemAdvise(vptr_[dim] + begin, size() * sizeof(float), cudaMemAdviseSetPreferredLocation, 0));
		CUDA_CHECK(cudaMemAdvise(vptr_[dim] + begin, size() * sizeof(float), cudaMemAdviseUnsetReadMostly, 0));
//		CUDA_CHECK(cudaMemPrefetchAsync(vptr_[dim] + begin, size * sizeof(float), 0, stream));
	}
	for (int dim = 0; dim < NDIM; dim++) {
		CUDA_CHECK(cudaMemAdvise(xptr_[dim] + begin, size() * sizeof(fixed32), cudaMemAdviseSetPreferredLocation, 0));
		CUDA_CHECK(cudaMemAdvise(xptr_[dim] + begin, size() * sizeof(fixed32), cudaMemAdviseSetReadMostly, 0));
//		CUDA_CHECK(cudaMemPrefetchAsync(xptr_[dim] + begin, size * sizeof(fixed32), 0, stream));
	}
	CUDA_CHECK(cudaMemAdvise(rptr_ + begin, size() * sizeof(rung_t), cudaMemAdviseSetPreferredLocation, 0));
//	CUDA_CHECK(cudaMemPrefetchAsync(rptr_ + begin, size * sizeof(rung_t), 0, stream));
}

void particle_set::prepare_drift(cudaStream_t stream) {
	return;
	for (int dim = 0; dim < NDIM; dim++) {
		CUDA_CHECK(cudaMemAdvise(vptr_[dim], size() * sizeof(float), cudaMemAdviseSetPreferredLocation, 0));
		CUDA_CHECK(cudaMemAdvise(vptr_[dim], size() * sizeof(float), cudaMemAdviseSetReadMostly, 0));
//		CUDA_CHECK(cudaMemPrefetchAsync(vptr_[dim], size() * sizeof(float), 0, stream));
	}
	for (int dim = 0; dim < NDIM; dim++) {
		CUDA_CHECK(cudaMemAdvise(xptr_[dim], size() * sizeof(fixed32), cudaMemAdviseSetPreferredLocation, 0));
		CUDA_CHECK(cudaMemAdvise(xptr_[dim], size() * sizeof(fixed32), cudaMemAdviseUnsetReadMostly, 0));
//		CUDA_CHECK(cudaMemPrefetchAsync(xptr_[dim], size() * sizeof(fixed32), 0, stream));
	}
}

void particle_set::prepare_sort() {
	return;
	int device = cudaCpuDeviceId;
//	auto stream = get_stream();
	for (int dim = 0; dim < NDIM; dim++) {
		CUDA_CHECK(cudaMemAdvise(vptr_[dim], size() * sizeof(float), cudaMemAdviseSetPreferredLocation, device));
		CUDA_CHECK(cudaMemAdvise(vptr_[dim], size() * sizeof(float), cudaMemAdviseUnsetReadMostly, device));
//		CUDA_CHECK(cudaMemPrefetchAsync(vptr_[dim], size() * sizeof(float), device, stream));
	}
	for (int dim = 0; dim < NDIM; dim++) {
		CUDA_CHECK(cudaMemAdvise(xptr_[dim], size() * sizeof(fixed32), cudaMemAdviseSetPreferredLocation, device));
		CUDA_CHECK(cudaMemAdvise(xptr_[dim], size() * sizeof(fixed32), cudaMemAdviseUnsetReadMostly, device));
//		CUDA_CHECK(cudaMemPrefetchAsync(xptr_[dim], size() * sizeof(fixed32), device, stream));
	}
	CUDA_CHECK(cudaMemAdvise(rptr_, size() * sizeof(rung_t), cudaMemAdviseSetPreferredLocation, device));
//	CUDA_CHECK(cudaMemAdvise(rptr_, size() * sizeof(rung_t), cudaMemAdviseUnsetReadMostly, device));
//	CUDA_CHECK(cudaMemPrefetchAsync(rptr_, size() * sizeof(rung_t), device, stream));
//	CUDA_CHECK(cudaStreamSynchronize(stream));
//	cleanup_stream(stream);
}

particle_set::particle_set(size_t size, size_t offset) {
	offset_ = offset;
	size_ = size;
	virtual_ = false;
	size_t chunk_size = NDIM * (sizeof(fixed32) + sizeof(float)) + sizeof(rung_t);
#ifdef TEST_FORCE
	chunk_size += (NDIM + 1) * sizeof(float);
#endif
	uint8_t *data;
	unified_allocator alloc;
	data = (uint8_t*) alloc.allocate(chunk_size * size);
	CHECK_POINTER(data);
	pptr_ = (particle*) data;
	for (size_t dim = 0; dim < NDIM; dim++) {
		xptr_[dim] = (fixed32*) (data + dim * size * sizeof(fixed32));
//		CUDA_CHECK(cudaMemAdvise(xptr_[dim], size * sizeof(fixed32), cudaMemAdviseSetReadMostly, 0));
//		CUDA_CHECK(cudaMemAdvise(xptr_[dim], size * sizeof(fixed32), cudaMemAdviseSetAccessedBy, 0));
	}
	for (size_t dim = 0; dim < NDIM; dim++) {
		vptr_[dim] = (float*) (data + size_t(NDIM) * size * sizeof(fixed32) + dim * size * sizeof(float));
//		CUDA_CHECK(cudaMemAdvise(vptr_[dim], size * sizeof(float), cudaMemAdviseSetAccessedBy, 0));
	}
	rptr_ = (rung_t*) (data + size_t(NDIM) * (sizeof(float) + sizeof(fixed32)) * size);
//	CUDA_CHECK(cudaMemAdvise(rptr_, size * sizeof(int8_t), cudaMemAdviseSetAccessedBy, 0));
#ifdef TEST_FORCE
	const auto offset1 = size_t(NDIM) * (sizeof(float) + sizeof(fixed32)) * size + sizeof(rung_t) * size;
	for (size_t dim = 0; dim < NDIM; dim++) {
		gptr_[dim] = (float*) (data + offset1 + dim * size * sizeof(float));
	}
	eptr_ = (float*) (data + offset1 + NDIM * size * sizeof(float));
#endif
	for (int i = 0; i < size; i++) {
		rptr_[i] = 0;
	}
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
		unified_allocator alloc;
		alloc.deallocate(pptr_);
	}
}

void particle_set::generate_random() {
	for (int i = 0; i < size_; i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			pos(dim, i) = rand_fixed32();
			vel(dim, i) = 0.f;
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
				pos(2, iii) = (i + 0.5) / dim;
				pos(1, iii) = (j + 0.5) / dim;
				pos(0, iii) = (k + 0.5) / dim;
				for (int dim = 0; dim < NDIM; dim++) {
					vel(dim, iii) = 0.f;
				}
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
std::vector<particle> load_particles(std::string filename);

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

	size_t lo = begin - offset_;
	size_t hi = end - offset_;
	fixed32 xmid(xm);
	fixed32* x = xptr_[xdim];
	while (lo < hi) {
		if (x[lo] >= xmid) {
			while (lo != hi) {
				hi--;
				if (x[hi] < xmid) {
					for (int dim = 0; dim < NDIM; dim++) {
						std::swap(xptr_[dim][hi], xptr_[dim][lo]);
					}
					for (int dim = 0; dim < NDIM; dim++) {
						std::swap(vptr_[dim][hi], vptr_[dim][lo]);
					}
					std::swap(rptr_[hi], rptr_[lo]);
					break;
				}
			}
		}
		lo++;
	}
	return hi + offset_;
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
		vel(0, i) = vx * std::pow(c0, 1.5);
		vel(1, i) = vy * std::pow(c0, 1.5);
		vel(2, i) = vz * std::pow(c0, 1.5);
		set_rung(0, i);
	}
	fread(&dummy, sizeof(dummy), 1, fp);
	fclose(fp);
}
