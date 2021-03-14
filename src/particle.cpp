#include <cosmictiger/particle.hpp>
#include <cosmictiger/memory.hpp>
#include <cosmictiger/rand.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/global.hpp>

#include <unordered_map>
#include <algorithm>

void particle_set::prepare_kick(cudaStream_t stream) {
	for (int dim = 0; dim < NDIM; dim++) {
		CUDA_CHECK(cudaMemAdvise(vptr_[dim], size() * sizeof(float), cudaMemAdviseSetPreferredLocation, 0));
		CUDA_CHECK(cudaMemAdvise(vptr_[dim], size() * sizeof(fixed32), cudaMemAdviseUnsetReadMostly, 0));
//		CUDA_CHECK(cudaMemAdvise(vptr_[dim], size() * sizeof(float), cudaMemAdviseSetAccessedBy, 0));
		CUDA_CHECK(cudaMemPrefetchAsync(vptr_[dim], size() * sizeof(float), 0, stream));
	}
	for (int dim = 0; dim < NDIM; dim++) {
		CUDA_CHECK(cudaMemAdvise(xptr_[dim], size() * sizeof(fixed32), cudaMemAdviseSetPreferredLocation, 0));
		CUDA_CHECK(cudaMemAdvise(xptr_[dim], size() * sizeof(fixed32), cudaMemAdviseSetReadMostly, 0));
//		CUDA_CHECK(cudaMemAdvise(xptr_[dim], size() * sizeof(fixed32), cudaMemAdviseSetAccessedBy, 0));
		CUDA_CHECK(cudaMemPrefetchAsync(xptr_[dim], size() * sizeof(fixed32), 0, stream));
	}
	CUDA_CHECK(cudaMemAdvise(rptr_, size() * sizeof(int8_t), cudaMemAdviseSetPreferredLocation, 0));
//	CUDA_CHECK(cudaMemAdvise(rptr_, size() * sizeof(int8_t), cudaMemAdviseSetAccessedBy, 0));
	CUDA_CHECK(cudaMemPrefetchAsync(rptr_, size() * sizeof(int8_t), 0, stream));
}

void particle_set::prepare_drift(cudaStream_t stream) {
	for (int dim = 0; dim < NDIM; dim++) {
		CUDA_CHECK(cudaMemAdvise(vptr_[dim], size() * sizeof(float), cudaMemAdviseSetPreferredLocation, 0));
		CUDA_CHECK(cudaMemAdvise(vptr_[dim], size() * sizeof(fixed32), cudaMemAdviseSetReadMostly, 0));
//		CUDA_CHECK(cudaMemAdvise(vptr_[dim], size() * sizeof(float), cudaMemAdviseSetAccessedBy, 0));
		CUDA_CHECK(cudaMemPrefetchAsync(vptr_[dim], size() * sizeof(float), 0, stream));
	}
	for (int dim = 0; dim < NDIM; dim++) {
		CUDA_CHECK(cudaMemAdvise(xptr_[dim], size() * sizeof(fixed32), cudaMemAdviseSetPreferredLocation, 0));
		CUDA_CHECK(cudaMemAdvise(xptr_[dim], size() * sizeof(fixed32), cudaMemAdviseUnsetReadMostly, 0));
		//	CUDA_CHECK(cudaMemAdvise(xptr_[dim], size() * sizeof(fixed32), cudaMemAdviseSetAccessedBy, 0));
		CUDA_CHECK(cudaMemPrefetchAsync(xptr_[dim], size() * sizeof(fixed32), 0, stream));
	}
}

void particle_set::prepare_sort(size_t begin, size_t end, int device) {
auto stream = get_stream();
	for (int dim = 0; dim < NDIM; dim++) {
		CUDA_CHECK(cudaMemAdvise(vptr_[dim] + begin, size() * sizeof(float), cudaMemAdviseSetPreferredLocation, device));
		CUDA_CHECK(cudaMemAdvise(vptr_[dim] + begin, size() * sizeof(float), cudaMemAdviseUnsetReadMostly, device));
		//	CUDA_CHECK(cudaMemAdvise(vptr_[dim] + begin, size() * sizeof(float), cudaMemAdviseSetAccessedBy, device));
	//	if (device != cudaCpuDeviceId) {
		CUDA_CHECK(cudaMemPrefetchAsync(vptr_[dim] + begin, size() * sizeof(float), device, stream));
	//	}
	}
	for (int dim = 0; dim < NDIM; dim++) {
		CUDA_CHECK(
				cudaMemAdvise(xptr_[dim] + begin, size() * sizeof(fixed32), cudaMemAdviseSetPreferredLocation, device));
		CUDA_CHECK(cudaMemAdvise(xptr_[dim] + begin, size() * sizeof(fixed32), cudaMemAdviseUnsetReadMostly, device));
		//	CUDA_CHECK(cudaMemAdvise(xptr_[dim] + begin, size() * sizeof(fixed32), cudaMemAdviseSetAccessedBy, device));
	//	if (device != cudaCpuDeviceId) {
			CUDA_CHECK(cudaMemPrefetchAsync(xptr_[dim] + begin, size() * sizeof(fixed32), device, stream));
	//	}
	}
	CUDA_CHECK(cudaMemAdvise(rptr_ + begin, size() * sizeof(int8_t), cudaMemAdviseSetPreferredLocation, device));
	CUDA_CHECK(cudaMemAdvise(rptr_ + begin, size() * sizeof(int8_t), cudaMemAdviseUnsetReadMostly, device));
//	CUDA_CHECK(cudaMemAdvise(rptr_ + begin, size() * sizeof(int8_t), cudaMemAdviseSetAccessedBy, device));
//	if (device != cudaCpuDeviceId) {
		CUDA_CHECK(cudaMemPrefetchAsync(rptr_ + begin, size() * sizeof(int8_t), device, stream));
	//}
	CUDA_CHECK(cudaStreamSynchronize(stream));
	cleanup_stream(stream);
}

particle_set::particle_set(size_t size, size_t offset) {
	offset_ = offset;
	size_ = size;
	virtual_ = false;
	size_t chunk_size = NDIM * (sizeof(fixed32) + sizeof(float)) + sizeof(uint8_t) + sizeof(uint64_t);
#ifdef TEST_FORCE
	chunk_size += (NDIM+1) * sizeof(float);
#endif
	uint8_t *data;
	CUDA_MALLOC(data, chunk_size * size);
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
	rptr_ = (int8_t*) (data + size_t(NDIM) * (sizeof(float) + sizeof(fixed32)) * size);
//	CUDA_CHECK(cudaMemAdvise(rptr_, size * sizeof(int8_t), cudaMemAdviseSetAccessedBy, 0));
	mptr_ = (uint64_t*) (data + size_t(NDIM) * (sizeof(float) + sizeof(fixed32)) * size + sizeof(int8_t) * size);
//	CUDA_CHECK(cudaMemAdvise(mptr_, size * sizeof(uint64_t), cudaMemAdviseSetAccessedBy, 0));
#ifdef TEST_FORCE
	const auto offset1 = size_t(NDIM) * (sizeof(float) + sizeof(fixed32)) * size + sizeof(int8_t) * size
	+ size * sizeof(uint64_t);
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
		CUDA_FREE(pptr_);
	}
}

void particle_set::generate_random() {
	for (int i = 0; i < size_; i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			pos(dim, i) = rand_fixed32();
			vel(dim, i) = 0.f;
		}
		set_rung(0, i);
		set_mid(0, i);
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
				set_mid(0, i);
			}
		}
	}
}

std::vector<size_t> particle_set::local_sort(size_t start, size_t stop, int64_t depth, morton_t key_min,
		morton_t key_max) {
	timer tm;
	std::vector<size_t> begin;
	std::vector<size_t> end;
	tm.start();
	if (stop - start >= MIN_CUDA_SORT) {
		timer tm;
		tm.start();
		auto stream = get_stream();
		//	prepare_sort1(stream);
		begin = cuda_keygen(*this, start, stop, depth, key_min, key_max, stream);
		//	prepare_sort2(stream);
		cleanup_stream(stream);
		//  printf( "%li %li\n", start , stop);
		size_t key_cnt = key_max - key_min;
		end.resize(key_cnt);
		for (size_t i = 0; i < key_max - key_min; i++) {
			end[i] = begin[i + 1];
		}
		assert(end[key_max - 1 - key_min] == stop);
		tm.stop();
		//  printf( "%e\n", (double)tm.read());
	} else {
		std::vector<int> counts(key_max - key_min, 0);
		for (size_t i = start; i < stop; i++) {
			const auto x = pos(i);
			const auto key = morton_key(x, depth);
			assert(key >= key_min);
			assert(key < key_max);
			counts[key - key_min]++;
			set_mid(key, i);
		}
		begin.resize(key_max - key_min + 1);
		end.resize(key_max - key_min + 1);
		begin[0] = start;
		end[0] = start + counts[0];
		for (int key = key_min + 1; key < key_max; key++) {
			const auto this_count = counts[key - key_min];
			const auto key_i = key - key_min;
			begin[key_i] = end[key_i - 1];
			end[key_i] = end[key_i - 1] + this_count;
		}
	}
	tm.stop();
	particle p;
	morton_t next_key;
	tm.reset();
	tm.start();
	std::atomic<int> sorted(0);
	for (morton_t first_key = key_min; first_key < key_max; first_key++) {
		bool flag = true;
		bool first = true;
		int64_t first_index;
		morton_t this_key = first_key;
		while (flag) {
			flag = false;
			for (size_t i = begin[this_key - key_min]++; i < end[this_key - key_min]; i = begin[this_key - key_min]++) {
				const auto x = pos(i);
				const int test_key = mid(i);
				if (test_key != this_key) {
					sorted++;
					flag = true;
					const auto tmp = p;
					p = part(i);
					if (!first) {
						for (int dim = 0; dim < NDIM; dim++) {
							pos(dim, i) = tmp.x[dim];
						}
						for (int dim = 0; dim < NDIM; dim++) {
							vel(dim, i) = tmp.v[dim];
						}
						set_rung(tmp.rung, i);
						set_mid(this_key, i);
					} else {
						first_index = i;
					}
					first = false;
					this_key = test_key;
					break;
				}
			}
			if (!flag && !first) {
				for (int dim = 0; dim < NDIM; dim++) {
					pos(dim, first_index) = p.x[dim];
				}
				for (int dim = 0; dim < NDIM; dim++) {
					vel(dim, first_index) = p.v[dim];
				}
				set_rung(p.rung, first_index);
				set_mid(this_key, first_index);
			}
		}
	}
	tm.stop();
//	printf("Sort took %e s, %i sorted.\n", tm.read(), sorted);
#ifdef TEST_RADIX
	bool failed = false;
	for (int i = start; i < stop - 1; i++) {
		if (mid(i + 1) < mid(i)) {
			printf("Radix failed : %lx %lx\n", mid(i), mid(i + 1));
			failed = true;
		}
	}
	if (failed) {
		abort();
	}
#endif
	//  printf( "bounds size = %li\n", bounds.size());
	begin.resize(0);
	begin.push_back(start);
	begin.insert(begin.end(), end.begin(), end.begin() + key_max - key_min);
	return begin;
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
