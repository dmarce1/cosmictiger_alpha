/*
 * particle.hpp
 *
 *  Created on: Jan 23, 2021
 *      Author: dmarce1
 */

#include <cosmictiger/particle.hpp>
#include <cosmictiger/memory.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/rand.hpp>

#include <algorithm>
#include <cassert>
#include <thread>

particle_set particle_set::parts;

HPX_PLAIN_ACTION(particle_set::generate_random_particle_set, generate_random_particle_set_action);
HPX_PLAIN_ACTION(particle_set::sort, sort_action);
HPX_PLAIN_ACTION(particle_set::get_count, get_count_action);
HPX_PLAIN_ACTION(particle_set::remote_sort, remote_sort_action);
HPX_PLAIN_ACTION(particle_set::get_sort_parts, get_sort_parts_action);

std::array<fixed32, NDIM> particle_set::get_x(size_t i) const {
   std::array<fixed32, NDIM> X;
   for (int dim = 0; dim < NDIM; dim++) {
      X[dim] = x[dim][i + offset];
//      printf( "%e\n", X[dim].to_float());
   }
   return X;
}

fixed32 particle_set::get_x(size_t i, int dim) const {
   return x[dim][i + offset];
}

particle_set particle_set::local_particle_set() {
   particle_set ps = parts;
   ps.virtual_ = false;
   return ps;
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
         fixed32 x;
         while ((x = rand_fixed32()) == fixed32(0)) {
            ;
         }
//         if (dim == 1) {
//            bool coin = i + parts.offset > global().opts.nparts / 2;
//            if (rand_float() < 0.01) {
//               coin = !coin;
//            }
//            if (coin) {
//               if (x < fixed32(0.0)) {
//                  x += fixed32(0.5);
//               }
//            } else {
//               if (x > fixed32(0.0)) {
//                  x -= fixed32(0.5);
//               }
//            }
//         }
         parts.x[dim][index] = x;
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

std::atomic<size_t> particle_set::hi_index;

std::vector<particle> particle_set::get_sort_parts(const std::vector<particle> &lo_parts, int xdim, fixed32 xmid) {

   const int nthread = hpx::thread::hardware_concurrency();
   std::vector<particle> hi_parts;
   std::vector<hpx::future<void>> futs;
   std::atomic < size_t > index;

   index = 0;
   hi_parts.reserve(lo_parts.size());
   futs.reserve(nthread - 1);
   const size_t hi = hi_index += lo_parts.size();

   const auto sort_func = [&](int i) {
      size_t this_hi, jb, je;
      jb = (size_t) i * lo_parts.size() / (size_t) nthread;
      je = (size_t)(i + 1) * lo_parts.size() / (size_t) nthread;
      this_hi = hi + i * lo_parts.size() / nthread;
      for (int j = jb; j < jb; j++) {
         do {
            this_hi++;
            assert(hi_index < parts.size);
         } while (parts.x[xdim][this_hi + parts.offset] <= xmid);
         particle p = lo_parts[j];
         const size_t i0 = this_hi + parts.offset;
         for (int dim = 0; dim < NDIM; dim++) {
            swap(p.x[dim], parts.x[dim][i0]);
         }
         for (int dim = 0; dim < NDIM; dim++) {
            std::swap(p.v[dim], parts.v[dim][i0]);
         }
         std::swap(p.rung, parts.rung[i0]);
         hi_parts[index++] = p;
      }
   };

   for (int i = 0; i < nthread; i++) {
      futs.push_back(hpx::async(sort_func, i));
   }
   sort_func(0);
   hpx::wait_all(futs.begin(), futs.end());
   return std::move(hi_parts);
}

std::vector<particle_set::count_t> particle_set::get_count(size_t begin, size_t end, int dim, fixed32 xmid) {
   const int nthread = hpx::thread::hardware_concurrency();
   hpx::future<std::vector<count_t>> count_left, count_right;
   std::vector<count_t> counts;
   std::vector<hpx::future<void>> futs;
   auto children = rel_children(begin, end);
   std::atomic<size_t> lo_cnt, hi_cnt;
   hpx::lcos::local::mutex mtx;
   count_t total_count;

   futs.reserve(nthread - 1);
   hi_index = 0;

   if (children.first != hpx::invalid_id) {
      count_left = hpx::async < get_count_action > (children.first, begin, end, dim, xmid);
   }
   if (children.second != hpx::invalid_id) {
      count_right = hpx::async < get_count_action > (children.second, begin, end, dim, xmid);
   }

   total_count.hi = total_count.lo = 0;
   const auto sort_func = [&](int j) {
      const size_t start = size_t(j) * parts.size / size_t(nthread);
      const size_t stop = size_t(j + 1) * parts.size / size_t(nthread);
      size_t myhi;
      size_t mylo;
      myhi = mylo = 0;
      for (int i = start; i < stop; i++) {
         if (parts.x[dim][i + parts.offset] > xmid) {
            myhi++;
         } else {
            mylo++;
         }
      }
      std::lock_guard < hpx::lcos::local::mutex > lock(mtx);
      total_count.hi += myhi;
      total_count.lo += mylo;
   };
   total_count.rank = hpx_rank();
   for (int i = 1; i < nthread; i++) {
      futs.push_back(hpx::async(sort_func, i));
   }
   sort_func(0);
   hpx::wait_all(futs.begin(), futs.end());
   counts.push_back(total_count);

   if (count_left.valid()) {
      const auto tmp = count_left.get();
      counts.insert(counts.end(), tmp.begin(), tmp.end());
   }
   if (count_right.valid()) {
      const auto tmp = count_left.get();
      counts.insert(counts.end(), tmp.begin(), tmp.end());
   }
   return std::move(counts);
}

size_t particle_set::remote_sort(std::vector<count_t> counts, size_t begin, size_t end, int xdim, fixed32 xmid) {
   const int nthread = hpx::thread::hardware_concurrency();
   const auto children = rel_children(begin, end);
   const auto &localities = hpx_localities();
   const int group_start = index_to_rank(begin);
   const int my_rel_rank = hpx_rank() - group_start;
   const int group_size = index_to_rank(end - 1) - group_start;
   const size_t set_size = end - begin;
   hpx::lcos::local::mutex mtx;
   hpx::future<void> sort_left, sort_right;

   if (children.first != hpx::invalid_id) {
      sort_left = hpx::async < remote_sort_action > (children.first, counts, begin, end, xdim, xmid);
   }
   if (children.second != hpx::invalid_id) {
      sort_right = hpx::async < remote_sort_action > (children.second, counts, begin, end, xdim, xmid);
   }

   std::sort(counts.begin(), counts.end());

   std::vector<std::pair<size_t, size_t>> hi_ranges, lo_ranges;
   std::vector<count_t> dests;
   int hi_cnt, lo_cnt, middle;
   std::pair<size_t, size_t> hi_range, lo_range;
   count_t dest;
   size_t mystop, mystart, this_start, this_stop;

   lo_cnt = 0;
   hi_ranges.reserve(group_size);
   lo_ranges.reserve(group_size);
   for (int i = 0; i < group_size; i++) {
      lo_cnt += counts[i].lo;
   }
   middle = lo_cnt;
   lo_cnt = 0;
   hi_cnt = 0;
   for (int i = 0; i < group_size; i++) {
      const auto range = rank_to_range(i + group_size);
      if (range.first < middle) {
         lo_range.first = lo_cnt;
         lo_cnt += counts[i].lo;
         lo_range.second = lo_cnt;
         lo_ranges.push_back(lo_range);
      }
      if (range.second >= middle) {
         hi_range.first = hi_cnt;
         hi_cnt += counts[i].hi;
         hi_range.second = hi_cnt;
         hi_ranges.push_back(hi_range);
      }
   }
   middle = lo_cnt;
   if (middle < -parts.offset + parts.size) {
      mystart = lo_ranges[my_rel_rank].first;
      mystop = lo_ranges[my_rel_rank].second;
      for (int i = 0; i < group_size; i++) {
         this_start = hi_ranges[i].first;
         this_stop = hi_ranges[i].second;
         dest.lo = std::max(mystart, this_start);
         dest.hi = std::min(mystop, this_stop);
         dest.rank = i + group_start;
         if (dest.hi > dest.lo) {
            dests.push_back(dest);
         }
      }
      std::vector < hpx::future<std::vector<particle>> > futs_out;
      std::vector < hpx::future<std::vector<particle>> > futs_in;
      std::vector<particle> all_sends;
      for (const auto dest : dests) {
         const auto sort_func = [&](int i) {
            std::vector<particle> send_parts;
            const size_t size = dest.hi - dest.lo;
            const size_t start = (size_t) i * size / (size_t) nthread;
            const size_t stop = (size_t)(i + 1) * size / (size_t) nthread;
            for (int j = start; j < stop; j++) {
               particle p;
               const size_t index = j + parts.offset;
               if (parts.x[xdim][index] > xmid) {
                  for (int dim = 0; dim < NDIM; dim++) {
                     p.x[dim] = parts.x[dim][index];
                  }
                  for (int dim = 0; dim < NDIM; dim++) {
                     p.v[dim] = parts.v[dim][index];
                  }
                  p.rung = parts.rung[index];
               }
               send_parts.push_back(p);
            }
            std::lock_guard < hpx::lcos::local::mutex > lock(mtx);
            all_sends.insert(all_sends.end(), send_parts.begin(), send_parts.end());
         };
         const auto locality = localities[group_start + dest.rank];
         for (int i = 1; i < nthread; i++) {
            futs_in.push_back(hpx::async < get_sort_parts_action > (locality, std::move(all_sends), xdim, xmid));
         }
      }
      std::vector<hpx::future<void>> futs;
      while (futs_in.size()) {
         hpx::wait_any(futs_in.begin(), futs_in.end());
         for (int i = 0; i < futs_in.size(); i++) {
            if (futs_in[i].is_ready()) {
               std::atomic < size_t > index(0);
               const auto tmp = futs_in[i].get();
               const auto sort_func = [&](int j) {
                  const size_t size = dests[j].hi - dests[j].lo;
                  const size_t start = (size_t) i * size / (size_t) nthread;
                  const size_t stop = (size_t)(i + 1) * size / (size_t) nthread;
                  for (int j = start; j < stop; j++) {
                     const int i1 = j + parts.offset;
                     if (parts.x[xdim][i1] > xmid) {
                        const int i2 = index++;
                        for (int dim = 0; dim < NDIM; dim++) {
                           parts.x[dim][i1] = tmp[i2].x[dim];
                        }
                        for (int dim = 0; dim < NDIM; dim++) {
                           parts.v[dim][i1] = tmp[i2].v[dim];
                        }
                        parts.rung[i1] = tmp[i2].rung;
                     }
                  }
               };
               for (int j = 0; j < nthread; j++) {
                  futs.push_back(hpx::async(sort_func, j));
               }
               std::lock_guard < hpx::lcos::local::mutex > lock(mtx);
               futs_in[i] = std::move(futs_in[futs_in.size() - 1]);
               futs_in.pop_back();
            }
         }
      }
      hpx::wait_all(futs.begin(), futs.begin());
   }
   if (sort_left.valid()) {
      sort_left.get();
   }
   if (sort_right.valid()) {
      sort_right.get();
   }
   return middle;
}

/* This function sorts particle ranges that are entirely contained on one locality
 *
 */
size_t particle_set::local_sort(size_t begin, size_t end, int xdim, fixed32 xmid) {
   int64_t hi, lo;
   hi = end - 1;
   lo = begin;
   //     printf( "%li %li\n", hi, lo);
   // go through list swapping particles hi and lo particles
   // until sorted

   const auto swap_parts = [](int i, int j) {
      const size_t i0 = i + parts.offset;
      const size_t j0 = j + parts.offset;
      for (int dim = 0; dim < NDIM; dim++) {
         swap(parts.x[dim][j0], parts.x[dim][i0]);
      }
      for (int dim = 0; dim < NDIM; dim++) {
         std::swap(parts.v[dim][j0], parts.v[dim][i0]);
      }
      std::swap(parts.rung[j0], parts.rung[i0]);
   };

   for (int i = lo; i <= hi; i++) {
      if (parts.x[xdim][i + parts.offset].to_float() == 0.0)
         printf("-------------------\n");
   }
   size_t mid = lo;
   if (hi - lo > 1) {
      while (lo != hi) {
         while (parts.x[xdim][lo + parts.offset] <= xmid && lo != hi) {
            lo++;
         }
         while (parts.x[xdim][hi + parts.offset] > xmid && lo != hi) {
            hi--;
         }
         if (hi != lo) {
            swap_parts(hi, lo);
         }
         mid = hi;
         if (lo != hi) {
            while (parts.x[xdim][hi + parts.offset] > xmid && lo != hi) {
               hi--;
            }
            while (parts.x[xdim][lo + parts.offset] <= xmid && lo != hi) {
               lo++;
            }
            if (hi != lo) {
               swap_parts(hi, lo);
            }
            mid = hi + 1;
         }
      }

   } else {
      if (parts.x[xdim][lo + parts.offset] < xmid) {
         hi++;
      }
      mid = hi;
   }
   return mid;
}

size_t particle_set::sort(size_t begin, size_t end, int dim, fixed32 xmid) {

   size_t mid;

   sort_action remote;
   const int myrank = hpx_rank();
   if (index_to_rank(begin) != myrank || index_to_rank(end - 1) != myrank) {
      if (begin < -parts.offset || begin >= -parts.offset + parts.size) {
         mid = remote(hpx_localities()[index_to_rank(begin)], begin, end, dim, xmid);
      } else {
         mid = remote_sort(std::vector<count_t>(), begin, end, dim, xmid);
      }
   } else {
      mid = local_sort(begin, end, dim, xmid);
   }
   return mid;
}

