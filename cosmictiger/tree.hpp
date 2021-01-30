#pragma once

#include <cosmictiger/defs.hpp>
#include <cosmictiger/particle.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/multipole.hpp>

#include <memory>

#define LEFT 0
#define RIGHT 1

class tree;

struct sort_params {
   range box;
   int depth;
   int radix_depth;
   std::shared_ptr<std::vector<size_t>> bounds;
   size_t key_begin;
   size_t key_end;
   std::shared_ptr<managed_allocator<tree>> allocator;
   std::shared_ptr<managed_allocator<sort_params>> params_alloc;
   template<class A>
   void serialization(A &&arc, unsigned) {
      arc & box;
      arc & depth;
      arc & radix_depth;
      arc & bounds;
      arc & key_begin;
      arc & key_end;
      /********* ADD ALLOCATOR SERIALIZATION *****************/
   }

   void set_root() {
      const auto opts = global().opts;
      for( int dim = 0; dim < NDIM; dim++) {
         box.begin[dim] = fixed64(0.f);
         box.end[dim] = fixed64(1.f);
      }
      depth = 0;
      bounds = std::make_shared < std::vector < size_t >> (2);
      radix_depth = 0;
      (*bounds)[0] = 0;
      (*bounds)[1] = opts.nparts;
      key_begin = 0;
      key_end = 1;
      allocator = std::make_shared<managed_allocator<tree>>();
      params_alloc = std::make_shared<managed_allocator<sort_params>>();
   }

   std::pair<size_t, size_t> get_bounds() const {
      std::pair < size_t, size_t > rc;
      rc.first = (*bounds)[key_begin];
      rc.second = (*bounds)[key_end];
      return rc;
   }

   std::array<sort_params, NCHILD> get_children() const {
      std::array<sort_params, NCHILD> child;
      for (int i = 0; i < NCHILD; i++) {
         child[i].radix_depth = radix_depth;
         child[i].bounds = bounds;
         child[i].depth = depth + 1;
         child[i].box = box;
         child[i].allocator = allocator;
         child[i].params_alloc = params_alloc;
       }
      int sort_dim = depth % NDIM;
      child[LEFT].box.end[sort_dim] = child[RIGHT].box.begin[sort_dim] = (fixed64(box.begin[sort_dim]) + fixed64(box.end[sort_dim]))
            / fixed64(2);
      child[LEFT].key_begin = key_begin;
      child[LEFT].key_end = child[RIGHT].key_begin = ((key_begin + key_end) >> 1);
      child[RIGHT].key_end = key_end;
      return child;
   }
};

struct tree_client {
   uintptr_t ptr;
   int rank;
   tree_client() {
      rank = -1;
   }
   bool operator==(const tree_client& other) const {
      return rank == other.rank && ptr == other.ptr;
   }
   template<class A>
   void serialization(A &&arc, unsigned) {
      arc & ptr;
      arc & rank;
   }
};

struct sort_return {
   tree_client client;
   template<class A>
   void serialization(A &&arc, unsigned) {
      arc * client;
   }
};

struct tree {

private:
   multi_source multi;
   float radius;
   std::array<tree_client,NCHILD> children;
   size_t part_begin;
   size_t part_end;
   int depth;
   std::shared_ptr<managed_allocator<tree>> allocator;
public:
   static particle_set* particles;
   static void set_particle_set(particle_set*);
   static hpx::future<sort_return> create_child(sort_params*);

   tree();
   sort_return sort(sort_params* = nullptr);

};
