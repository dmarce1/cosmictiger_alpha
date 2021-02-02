#pragma once

#include <cosmictiger/defs.hpp>
#include <cosmictiger/containers.hpp>
#include <cosmictiger/particle.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/multipole.hpp>
#include <cosmictiger/fast_future.hpp>
#include <cosmictiger/expansion.hpp>

#include <memory>

#define LEFT 0
#define RIGHT 1

class tree;
struct sort_params;
struct tree_client;

#ifndef __CUDACC__
struct tree_alloc {
   managed_allocator<multipole> multi_alloc;
   managed_allocator<tree> tree_alloc;
};

struct sort_params {
#ifdef TEST_STACK
   uint8_t* stack_ptr;
#endif
   range box;
   std::shared_ptr<std::vector<size_t>> bounds;
   std::shared_ptr<tree_alloc> allocs;
   uint32_t key_begin;
   uint32_t key_end;
   int8_t depth;

   template<class A>
   void serialization(A &&arc, unsigned) {
      /********* ADD******/
   }

   sort_params() {
      depth = -1;
   }

   bool iamroot() const {
      return depth == -1;
   }

   void set_root() {
      const auto opts = global().opts;
      for (int dim = 0; dim < NDIM; dim++) {
         box.begin[dim] = fixed64(0.f);
         box.end[dim] = fixed64(1.f);
      }
#ifdef TEST_STACK
      stack_ptr = (uint8_t*) &stack_ptr;
#endif
      depth = 0;
      bounds = std::make_shared < std::vector < size_t >> (2);
      (*bounds)[0] = 0;
      (*bounds)[1] = opts.nparts;
      key_begin = 0;
      key_end = 1;
      allocs = std::make_shared<tree_alloc>();
   }

   std::pair<size_t, size_t> get_bounds() const {
      std::pair<size_t, size_t> rc;
      rc.first = (*bounds)[key_begin];
      rc.second = (*bounds)[key_end];
      return rc;
   }

   std::array<sort_params, NCHILD> get_children() const {
      std::array<sort_params, NCHILD> child;
      for (int i = 0; i < NCHILD; i++) {
         child[i].bounds = bounds;
         child[i].depth = depth + 1;
         child[i].allocs = allocs;
         child[i].box = box;
#ifdef TEST_STACK
         child[i].stack_ptr = stack_ptr;
#endif
      }
      int sort_dim = depth % NDIM;
      child[LEFT].box.end[sort_dim] = child[RIGHT].box.begin[sort_dim] = (fixed64(box.begin[sort_dim])
            + fixed64(box.end[sort_dim])) / fixed64(2);
      child[LEFT].key_begin = key_begin;
      child[LEFT].key_end = child[RIGHT].key_begin = ((key_begin + key_end) >> 1);
      child[RIGHT].key_end = key_end;
      return child;
   }
};
#endif

struct tree_client {
   uintptr_t ptr;
   int rank;

   CUDA_EXPORT
   constexpr tree_client() :
         rank(-1), ptr(0) {
   }
   CUDA_EXPORT
   bool operator==(const tree_client &other) const {
      return rank == other.rank && ptr == other.ptr;
   }
   CUDA_EXPORT
   inline operator tree*() {
      return reinterpret_cast<tree*>(ptr);
   }
   CUDA_EXPORT
   inline operator const tree*() {
      return reinterpret_cast<const tree*>(ptr);
   }
   template<class A>
   void serialization(A &&arc, unsigned) {
      arc & ptr;
      arc & rank;
   }
};

#ifndef __CUDACC__
struct sort_return {
   tree_client check;
   template<class A>
   void serialization(A &&arc, unsigned) {
      assert(false);
   }
};

#endif

struct call_stack_entry {
   vector<tree_client> dchecks;
   vector<tree_client> echecks;
   tree* tptr;
   array<exp_real, NDIM> Lcom;
   expansion L;
};

struct kick_params {
   vector<call_stack_entry> call_stack;
   vector<multipole*> multi_i;
   vector<pair<size_t, size_t>> part_i;
};

struct kick_return {
   int8_t rung;
};

struct tree {
#ifndef __CUDACC__
private:
#endif
   array<fixed32, NDIM> pos;
   float radius;
   array<tree_client, NCHILD> children;
   pair<size_t, size_t> parts;
   multipole *multi;
public:
   CUDA_EXPORT
   inline bool leaf() const {
      return children[0] == tree_client();
   }
   static kick_return kick(tree_client root_ptr, int rung, float theta);
#ifndef __CUDACC__
   static particle_set *particles;
   static void set_particle_set(particle_set*);
   inline static fast_future<sort_return> create_child(sort_params&);
   static fast_future<sort_return> cleanup_child();
   static kick_return step(int rung, float theta);
   tree();
   sort_return sort(sort_params = sort_params());
#endif
};
