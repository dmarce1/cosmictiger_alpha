/*
 * particle_sort.cpp
 *
 *  Created on: Jan 25, 2021
 *      Author: dmarce1
 */

/*
 * particle_source.cpp
 *
 *  Created on: Jan 25, 2021
 *      Author: dmarce1
 */

#include "../cosmictiger/particle_sort.hpp"

HPX_PLAIN_ACTION(particle_sort::sort, sort_action);
HPX_PLAIN_ACTION(particle_sort::get_count, get_count_action);
HPX_PLAIN_ACTION(particle_sort::remote_sort, remote_sort_action);
HPX_PLAIN_ACTION(particle_sort::get_sort_parts, get_sort_parts_action);

std::atomic<size_t> particle_sort::hi_index;

std::vector<particle> particle_sort::get_sort_parts(const std::vector<particle> &lo_parts, int xdim, fixed32 xmid) {

   const int nthread = std::thread::hardware_concurrency();
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
         } while (parts.x[xdim][this_hi] <= xmid);
         particle p = lo_parts[j];
         for (int dim = 0; dim < NDIM; dim++) {
            swap(p.x[dim], parts.x[dim][this_hi]);
         }
         for (int dim = 0; dim < NDIM; dim++) {
            std::swap(p.v[dim], parts.v[dim][this_hi]);
         }
         std::swap(p.rung, parts.rung[this_hi]);
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

std::vector<particle_sort::count_t> particle_sort::get_count(size_t begin, size_t end, int dim, fixed32 xmid) {
   const int nthread = std::thread::hardware_concurrency();
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

size_t particle_sort::remote_sort(std::vector<count_t> counts, size_t begin, size_t end, int xdim, fixed32 xmid) {
   const int nthread = std::thread::hardware_concurrency();
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
size_t particle_sort::local_sort(size_t begin, size_t end, int xdim, fixed32 xmid) {

   // struc for sorting bins
   struct bin_t {
      size_t begin, middle, end;
   };

   const int nthread = std::thread::hardware_concurrency();
   static std::atomic<int> used_threads(0);

   // compute numer of threads for this call
   int num_threads = 0;
   used_threads++;
   while (used_threads++ < nthread) {
      num_threads++;
   }

   // function to swap the data for two particles
   const auto swap_parts = [](int i, int j) {
      for (int dim = 0; dim < NDIM; dim++) {
         swap(parts.x[dim][j], parts.x[dim][i]);
      }
      for (int dim = 0; dim < NDIM; dim++) {
         std::swap(parts.v[dim][j], parts.v[dim][i]);
      }
      std::swap(parts.rung[j], parts.rung[i]);
   };

   const int numparts = end - begin;
   std::vector<bin_t> sort_bins(num_threads);
   std::vector<bin_t> next_sort_bins(num_threads);
   std::vector<hpx::future<void>> ifuts;

   // Function for the initial binned sort
   const auto initial_sort_func = [&](int j) {
      // Comute start and stop points for this thread
      const int start = (size_t)j * numparts / (size_t)num_threads;
      const int stop =(size_t) (j + 1) * numparts /(size_t) num_threads;
      // Iterate through this section;
      size_t hi, lo;
     for (int start = 0; start < stop; start++) {
         // start at top and bottom of list
         hi = stop - 1;
         lo = start;
         // go through list swapping particles hi and lo particles
         // until sorted
         while (lo < hi) {
            if (parts.x[xdim][lo] > xmid) {
               // find a particle to swap
               while (parts.x[xdim][hi] > xmid) {
                  hi--;
                  swap_parts(hi, lo);
               }
            }
            lo++;
         }
      }
      // Set sort bins data for the next stage of sort
      sort_bins[j].begin = start;
      sort_bins[j].end = stop;
      sort_bins[j].middle = lo;
   };
   // Spawn threads for first sort stage
   ifuts.reserve(num_threads - 1);
   for (int i = 1; i < num_threads; i++) {
      ifuts.push_back(hpx::async(initial_sort_func, i));
   }
   // Work for thread = 0 and wait for rest
   initial_sort_func(0);
   hpx::wait_all(ifuts.begin(), ifuts.end());

   // number of threads per section for next stage
   int this_num_threads = 1;

   // merge pairs of sort bins
   const auto merge_sort_pairs = [&](int i0, int i1, int j) {
      const auto &bin0 = sort_bins[i0];
      const auto &bin1 = sort_bins[i1];
      // number of lo particles in hi bin
      const size_t topcnt = bin1.middle - bin1.begin;
      // number of hi partiles in lo bin
      const size_t botcnt = bin0.end - bin0.middle;
      // Compute start and stop for the entire set
      size_t lo = bin0.middle;
      size_t hi = std::min(bin0.middle + topcnt, bin0.middle + botcnt);
      // compute start and stop for this set
      const size_t start =(size_t) j * (hi - lo) / (size_t)this_num_threads;
      const size_t stop = (size_t)(j + 1) * (hi - lo) / (size_t)this_num_threads;
      const size_t lo_start = lo + start;
      const size_t lo_stop = lo + stop;
      // index for first hi particle to be moved
      hi = bin1.middle - (stop - start);
      // go through list swapping particles from hi particles in the lo bin with
      // lo particles top hi bin
      for (lo = lo_start; lo < lo_stop; lo++) {
         swap_parts(lo, hi);
         hi++;
      }
      // merge sort bins
      next_sort_bins[i0 / 2].begin = bin0.begin;
      next_sort_bins[i0 / 2].end = bin1.end;
      next_sort_bins[i0 / 2].middle = lo + bin0.middle - bin0.begin + topcnt;

   };

   // We only need to reduce the bins if there is more than 1
   if (num_threads > 1) {
      // We are merging pairs of bins, so each merge set is half
      // as big as the last with accounting for odd numbers
      for (int P = num_threads / 2; P >= 1; P = (P + 1) / 2) {
         ifuts.resize(0);
         next_sort_bins.resize(this_num_threads);
        // number of threads per swap pair
         this_num_threads = std::min((int) num_threads, this_num_threads * 2);
         // Iterate on pairs
         for (int i = 0; i < P; i += 2) {
            // If this has a partner, merge them
            if (i + 1 < P) {
               for (int j = 0; j < num_threads; j++) {
                  ifuts.push_back(hpx::async(merge_sort_pairs, i, i + 1, j));
               }
               // if it doesn't have a partner push it to next list
            } else {
               next_sort_bins[i / 2] = sort_bins[i];
            }
         }
         // Wait for work then prepare for next iteration
         hpx::wait_all(ifuts.begin(), ifuts.end());
         sort_bins = std::move(next_sort_bins);
      }
   }
   // free these threads
   used_threads -= num_threads;
   return sort_bins[0].middle;
}

size_t particle_sort::sort(size_t begin, size_t end, int dim, fixed32 xmid) {
   sort_action remote;
   size_t mid;
   const int myrank = hpx_rank();
   if (index_to_rank(begin) != myrank || index_to_rank(end) != myrank) {
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

