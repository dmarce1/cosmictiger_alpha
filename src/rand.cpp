/*
 * rand.cpp
 *
 *  Created on: Jan 25, 2021
 *      Author: dmarce1
 */

#include <cosmictiger/rand.hpp>
#include <cosmictiger/hpx.hpp>

constexpr size_t a = 1664525;
constexpr size_t c = 1013904223;
constexpr size_t mod = ((size_t) 1 << (size_t) 32);

static hpx::lcos::local::mutex mtx;
static size_t number = 1234;

fixed32 rand_fixed32() {
   std::lock_guard < hpx::lcos::local::mutex > lock(mtx);
   fixed32 result;
   number = (a * number + c) % mod;
   result.i = number;
   return result;
}

size_t rand_size_t() {
   std::lock_guard < hpx::lcos::local::mutex > lock(mtx);
   number = (a * number + c) % mod;
   return number;
}

float rand_float() {
   std::lock_guard < hpx::lcos::local::mutex > lock(mtx);
   number = (a * number + c) % mod;
   return (number + 0.5) / mod;
}

