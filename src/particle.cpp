#include <cosmictiger/particle.hpp>
#include <cosmictiger/memory.hpp>
#include <cosmictiger/rand.hpp>
#include <cosmictiger/timer.hpp>

#include <unordered_map>

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
   rptr_ = (int*) (data + size_t(NDIM) * (sizeof(float) + sizeof(fixed32)) * size);
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
      rung(i) = 0;
   }
}

std::vector<size_t> particle_set::local_sort(size_t start, size_t stop, int64_t depth) {
   std::unordered_map < size_t, size_t > counts;
   std::vector < size_t > begin;
   std::vector < size_t > end;
   std::vector < size_t > keys;
   size_t key_max = 0;
   size_t key_min = ~(1 << (depth + 1));

   timer tm;
   tm.start();
   keys.reserve(stop - start);
   for (size_t i = start; i < stop; i++) {
      const auto x = pos(i);
      const auto key = morton_key(x, depth);
      auto iter = counts.find(key);
      if (iter != counts.end()) {
         iter->second++;
      } else {
         counts[key] = 1;
      }
      keys.push_back(key);
      key_max = std::max(key_max, key);
      key_min = std::min(key_min, key);
   }
   tm.stop();
   size_t key_cnt = key_max - key_min;
   begin.resize(key_cnt);
   end.resize(key_cnt + 1);
   begin[0] = start;
   end[0] = start + counts[key_min];
   for (int key = key_min + 1; key < key_max; key++) {
      const auto this_count = counts[key];
      const auto key_i = key - key_min;
      end[key_i] = end[key_i - 1] + this_count;
      begin[key_i] = end[key_i - 1];
   }
   printf("Key generation and count took %e s\n", tm.read());

   particle p;
   morton_t next_key;
   tm.reset();
   tm.start();
   for (morton_t first_key = key_min; first_key < key_max; first_key++) {
      bool flag = true;
      bool first = true;
      int64_t first_index = -1;
      morton_t this_key = first_key;
      while (flag) {
         flag = false;
         size_t i = begin[this_key - key_min]++;
         for (; i < end[this_key - key_min]; i = begin[this_key - key_min]++) {
            const auto x = pos(i);
            const int test_key = keys[i - start];
            if (test_key != this_key) {
               flag = true;
               const auto tmp = p;
               p = part(i);
               if (!first) {
                  part(i) = tmp;
                  std::swap(this_key, keys[i - start]);
               } else {
                  first_index = i;
               }
               first = false;
               this_key = test_key;
               break;
            }
         }
         if (!flag && !first) {
            part(first_index) = p;
            keys[first_index - start] = this_key;
         }
      }
   }
   tm.stop();
   printf("Sort took %e s\n", tm.read());

   return end;
}

