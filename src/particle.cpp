/*
 * particle.hpp
 *
 *  Created on: Jan 23, 2021
 *      Author: dmarce1
 */

#include <cosmictiger/global.hpp>
#include <cosmictiger/particle.hpp>
#include <cosmictiger/memory.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/rand.hpp>

#include <algorithm>
#include <cassert>
#include <thread>

particle_set::members_t particle_set::parts;

HPX_PLAIN_ACTION(particle_set::generate_random_particle_set, generate_random_particle_set_action);

void particle_set::create() {
   const auto nranks = hpx_localities().size();
   const auto &myrank = hpx_rank();
   const size_t &nparts = global().opts.nparts;
   const size_t mystart = myrank * nparts / nranks;
   const size_t mystop = (myrank + 1) * nparts / nranks;
   parts.offset = -mystart;
   parts.size = mystop - mystart;
   parts.virtual_ = false;
   for (int dim = 0; dim < NDIM; dim++) {
      CUDA_MALLOC(parts.x[dim], nparts);
      CUDA_MALLOC(parts.v[dim], nparts);
   }
   CUDA_MALLOC(parts.rung, nparts);
   parts.size = nparts;
}

void particle_set::destroy() {
   for (int dim = 0; dim < NDIM; dim++) {
      CUDA_FREE(parts.x[dim]);
      CUDA_FREE(parts.v[dim]);
   }
   CUDA_FREE(parts.rung);
}

int particle_set::index_to_rank(size_t index) {
   const int result = ((index + (size_t) 1) * (size_t) hpx_size()) / (global().opts.nparts + (size_t) 1);
   assert(index >= rank_to_range(result).first);
   assert(index < rank_to_range(result).second);
   return result;
}

std::pair<size_t, size_t> particle_set::rank_to_range(int rank) {
   const size_t nparts = global().opts.nparts;
   const size_t nranks = hpx_size();
   std::pair < size_t, size_t > rc;
   rc.first = (size_t) rank * nparts / (size_t) nranks;
   rc.second = (size_t)(rank + 1) * nparts / (size_t) nranks;
   return rc;
}

std::pair<hpx::id_type, hpx::id_type> particle_set::rel_children(size_t begin, size_t end) {
   const auto &localities = hpx_localities();
   const int first_rank = index_to_rank(begin);
   const int last_rank = index_to_rank(end - 1);
   const int n_ranks = last_rank - first_rank + 1;
   const int my_rel_rank = hpx_rank() - first_rank;
   const int my_rel_left = ((my_rel_rank + 1) << 1) - 1;
   const int my_rel_right = ((my_rel_rank + 1) << 1);

   std::pair < hpx::id_type, hpx::id_type > rc;
   if (my_rel_left < n_ranks) {
      rc.first = localities[my_rel_left + first_rank];
   }
   if (my_rel_right < n_ranks) {
      rc.second = localities[my_rel_right + first_rank];
   }
   return rc;
}

void particle_set::generate_random_particle_set() {
   const auto mychildren = hpx_child_localities();
   hpx::future<void> left, right;
   if (mychildren.first != hpx::invalid_id) {
      left = hpx::async < generate_random_particle_set_action > (mychildren.first);
   }
   if (mychildren.first != hpx::invalid_id) {
      right = hpx::async < generate_random_particle_set_action > (mychildren.second);
   }
   for (size_t i = 0; i < parts.size; i++) {
      size_t index = i + parts.offset;
      for (int dim = 0; dim < NDIM; dim++) {
         parts.x[dim][index] = rand_fixed32();
      }
      for (int dim = 0; dim < NDIM; dim++) {
         parts.v[dim][index] = 0.f;
      }
      parts.rung[index] = 0;
   }

   if (left.valid()) {
      left.get();
   }
   if (right.valid()) {
      right.get();
   }

}

