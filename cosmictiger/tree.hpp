#pragma once

#include <cosmictiger/defs.hpp>
#include <cosmictiger/particle.hpp>
#include <cosmictiger/hpx.hpp>

#include <memory>

#define LEFT 0
#define RIGHT 1

struct sort_params {
   range box;
   int depth;
   int radix_depth;
   std::shared_ptr<std::vector<size_t>> bounds;
   size_t key_begin;
   size_t key_end;
   template<class A>
   void serialization(A &&arc, unsigned) {
      arc & box;
      arc & depth;
      arc & radix_depth;
      arc & bounds;
      arc & key_begin;
      arc & key_end;
   }

   void set_root() {
      const auto opts = global().opts;
      for( int dim = 0; dim < NDIM; dim++) {
         box.begin[dim] = fixed32::min();
         box.end[dim] = fixed32::max();
      }
      depth = 0;
      bounds = std::make_shared < std::vector < size_t >> (2);
      radix_depth = 0;
      (*bounds)[0] = 0;
      (*bounds)[1] = opts.nparts;
      key_begin = 0;
      key_end = 1;
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
   std::array<tree_client,NCHILD> children;
   size_t part_begin;
   size_t part_end;
   int depth;

public:
   static particle_set* particles;
   static void set_particle_set(particle_set*);
   static hpx::future<sort_return> create_child(std::shared_ptr<sort_params>);

   tree();
   sort_return sort(std::shared_ptr<sort_params> = nullptr);

};
