#include <cosmictiger/particle.hpp>
#include <cosmictiger/memory.hpp>
#include <cosmictiger/rand.hpp>
#include <cosmictiger/timer.hpp>

#include <unordered_map>
#include <algorithm>

particle_set::particle_set(size_t size, size_t offset) {
   offset_ = offset;
   size_ = size;
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
   rptr_ = (particle::flags_t*) (data + size_t(NDIM) * (sizeof(float) + sizeof(fixed32)) * size);
}

particle_set::~particle_set() {
   CUDA_FREE(pptr_);
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

std::vector<size_t> particle_set::local_sort(size_t start, size_t stop, int64_t depth, morton_t key_min,
      morton_t key_max) {
   timer tm;
   tm.start();
   const auto key_min0 = key_min;
   const auto key_max0 = key_max;
   const auto bounds = cuda_keygen(*this, start, stop, depth, key_min, key_max);
   //  printf( "%li %li\n", start , stop);
   size_t key_cnt = key_max - key_min;
   std::vector < size_t > end;
   std::vector < size_t > begin;
   begin.reserve(key_cnt);
   end.resize(key_cnt);
   for (size_t i = 0; i < key_min - key_min0; i++) {
      begin.push_back(bounds[0]);
   }
   begin.insert(begin.end(), bounds.begin(), bounds.begin() + key_max + 1 - key_min);
   for (size_t i = key_max + 1 - key_min0; i <= key_max0 - key_min0; i++) {
      begin.push_back(bounds[key_max - key_min0]);
   }
   for (size_t i = 0; i < key_max0 - key_min0; i++) {
      end[i] = begin[i + 1];
   }
   printf("-----------\n");
   for (int i = 0; i < key_max0 - key_min0; i++) {
      printf(" %li %li %li\n", begin[i], end[i], bounds[i]);
   }
   printf("-----------\n");
   tm.stop();
   //  abort();
   //  printf("%li %li\n", end[key_max - 1 - key_min], stop - start);
   printf("-- %li %li %li %li\n", key_min0, key_min, key_max, key_max0);
 //  if (end[key_max - 1 - key_min] != stop - start) {
      printf("-- %li %li \n", end[key_max - 1 - key_min0], stop);
      assert(end[key_max - 1 - key_min0] == stop);
  // }
   printf("Key generation and count took %e s\n", tm.read());

   particle p;
   morton_t next_key;
   tm.reset();
   tm.start();
   int sorted = 0;
   for (morton_t first_key = key_min; first_key < key_max; first_key++) {
      bool flag = true;
      bool first = true;
      int64_t first_index;
      morton_t this_key = first_key;
      while (flag) {
         flag = false;
         for (size_t i = begin[this_key - key_min0]++; i < end[this_key - key_min0]; i = begin[this_key - key_min0]++) {
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
                  set_rung(tmp.flags.rung, i);
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
            set_rung(p.flags.rung, first_index);
            set_mid(this_key, first_index);
         }
      }
   }
   tm.stop();
   printf("Sort took %e s, %i sorted.\n", tm.read(), sorted);
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
   printf( "bounds size = %li\n", bounds.size());
   return bounds;
}

