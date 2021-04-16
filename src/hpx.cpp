/*
 * hpx.cpp
 *
 *  Created on: Jan 24, 2021
 *      Author: dmarce1
 */

#include <cosmictiger/hpx.hpp>

#include <vector>

static int myrank;
static int hpx_size_;
static std::vector<hpx::id_type> localities;
static std::pair<hpx::id_type, hpx::id_type> mychildren;

int hardware_concurrency() {
	return hpx::thread::hardware_concurrency();
}

void hpx_init() {
   int left, right;
   localities = hpx::find_all_localities();
   hpx_size_ = localities.size();
   myrank = hpx::get_locality_id();
   left = ((myrank + 1) << 1) - 1;
   right = ((myrank + 1) << 1);
   hpx_size_ = localities.size();
   if (left < hpx_size_) {
      mychildren.first = localities[left];
   }
   if (right < hpx_size_) {
      mychildren.first = localities[right];
   }
}

int hpx_size() {
   return hpx_size_;
}

int hpx_rank() {
   static const int rank = hpx::get_locality_id();
   return rank;
}


void hpx_yield() {
	hpx::this_thread::yield();
}

const std::vector<hpx::id_type>& hpx_localities() {
   return localities;
}

const std::pair<hpx::id_type, hpx::id_type>& hpx_child_localities() {
   return mychildren;
}
