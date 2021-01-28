#include <cosmictiger/particle.hpp>
#include <cosmictiger/memory.hpp>

#define CHUNK_SIZE (7)

particle_set::particle_set(size_t size, size_t offset) {
   offset_ = offset;
   size_ = size;
   size = (((size - 1) / CHUNK_SIZE) + 1) * CHUNK_SIZE;
   printf("%i %i\n", size, size_);
   format_ = format::soa;
   CUDA_MALLOC(pptr_, size);
   CHECK_POINTER(pptr_);
   int8_t *data = (int8_t*) pptr_;
   for (size_t dim = 0; dim < NDIM; dim++) {
      xptr_[dim] = (fixed32*) (data + dim * size * sizeof(fixed32));
   }
   for (size_t dim = 0; dim < NDIM; dim++) {
      vptr_[dim] = (float*) (data + size_t(NDIM) * size * sizeof(fixed32) + dim * size * sizeof(float));
   }
   rptr_ = (int*) (data + size_t(NDIM) * (sizeof(float) + sizeof(fixed32)) * size);
}

particle_set::~particle_set() {
   CUDA_FREE(pptr_);
}

void particle_set::set_mem_format(format new_format) {
   std::vector<morton_t> keys;
   const size_t smax = ((size_ - 1) / CHUNK_SIZE + 1) * CHUNK_SIZE;
   keys.reserve(smax);
   std::vector<particle> parts(CHUNK_SIZE);
   std::array<fixed32, NDIM> x;
   if (new_format != format_) {
      if (new_format == format::soa) {
         for (size_t i = 0; i < size_; i += CHUNK_SIZE) {
            for (size_t j = i; j < i + CHUNK_SIZE; j++) {
               for (size_t dim = 0; dim < NDIM; dim++) {
                  parts[dim] = *((particle*) (xptr_[dim]) + j / CHUNK_SIZE);
               }
               for (size_t dim = 0; dim < NDIM; dim++) {
                  parts[NDIM + dim] = *((particle*) (vptr_[dim]) + j / CHUNK_SIZE);
               }
               parts[2 * NDIM] = *((particle*) (rptr_) + j / CHUNK_SIZE);
            }
            for (size_t j = i; j < i + CHUNK_SIZE; j++) {
               const auto jmi = j - i;
               for (int dim = 0; dim < NDIM; dim++) {
                  xptr_[dim][j] = parts[jmi].x[dim];
               }
               for (int dim = 0; dim < NDIM; dim++) {
                  vptr_[dim][j] = parts[jmi].v[dim];
               }
               rptr_[j] = parts[jmi].rung;
            }
         }
      } else {
         for (size_t i = 0; i < size_; i += CHUNK_SIZE) {
            for (size_t j = i; j < i + CHUNK_SIZE; j++) {
               const auto jmi = j - i;
               for (int dim = 0; dim < NDIM; dim++) {
                  x[dim] = xptr_[dim][j];
                  parts[jmi].x[dim] = x[dim];
               }
               keys.push_back(morton_key<45>(x));
               for (int dim = 0; dim < NDIM; dim++) {
                  parts[jmi].v[dim] = vptr_[dim][j];
               }
               parts[jmi].rung = rptr_[j];
            }
            for (size_t j = i; j < i + CHUNK_SIZE; j++) {
               for (size_t dim = 0; dim < NDIM; dim++) {
                  *((particle*) (xptr_[dim]) + j / CHUNK_SIZE) = parts[dim];
               }
               for (size_t dim = 0; dim < NDIM; dim++) {
                  *((particle*) (vptr_[dim]) + j / CHUNK_SIZE) = parts[NDIM + dim];
               }
               *((particle*) (rptr_) + j / CHUNK_SIZE) = parts[2 * NDIM];
            }
         }

      }
      format_ = new_format;
   }
}
