#pragma once

#include <cosmictiger/defs.hpp>
#include <cosmictiger/particle.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/multipole.hpp>
#include <cosmictiger/fast_future.hpp>
#include <cosmictiger/expansion.hpp>
#include <cosmictiger/finite_vector.hpp>

#include <memory>
#include <stack>

#define LEFT 0
#define RIGHT 1
#define WORKSPACE_SIZE 1024

class tree;
struct sort_params;
struct tree_ptr;

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

struct kick_return {
   int8_t rung;
};

class tree_ptr;

using checks_type = finite_vector<tree_ptr,WORKSPACE_SIZE>;

struct kick_workspace_t {
   checks_type multi_interactions;
   checks_type part_interactions;
   checks_type next_checks;
};

struct kick_stack {
   finite_vector<finite_vector<tree_ptr, WORKSPACE_SIZE>, TREE_MAX_DEPTH> dchecks;
   finite_vector<finite_vector<tree_ptr, WORKSPACE_SIZE>, TREE_MAX_DEPTH> echecks;
};


struct tree_ptr {
   uintptr_t ptr;
   int rank;
#ifndef NDEBUG
   int constructed;
#endif
   tree_ptr() {
      rank = -1;
      ptr = 0;
#ifndef NDEBUG
      constructed = 1234;
#endif
   }
   tree_ptr(tree_ptr &&other) {
      rank = other.rank;
      ptr = other.ptr;
#ifndef NDEBUG
      constructed = 1234;
#endif
   }
   tree_ptr(const tree_ptr &other) {
      rank = other.rank;
      ptr = other.ptr;
#ifndef NDEBUG
      constructed = 1234;
#endif
   }
   tree_ptr& operator=(const tree_ptr &other) {
      assert(constructed == 1234);
      ptr = other.ptr;
      rank = other.rank;
      return *this;
   }
   tree_ptr& operator=(tree_ptr &&other) {
      assert(constructed == 1234);
      ptr = other.ptr;
      rank = other.rank;
      return *this;
   }
   bool operator==(const tree_ptr &other) const {
      assert(constructed == 1234);
      return rank == other.rank && ptr == other.ptr;
   }
   template<class A>
   void serialization(A &&arc, unsigned) {
      arc & ptr;
      arc & rank;
   }
   operator tree*() {
      assert(constructed == 1234);
      assert(ptr);
      return (tree*) (ptr);
   }
   operator const tree*() const {
      assert(constructed == 1234);
      assert(ptr);
      return (const tree*) (ptr);
   }
   fast_future<array<tree_ptr, NCHILD>> get_children() const;
   float get_radius() const;
   array<fixed32, NDIM> get_pos() const;
   bool is_leaf() const;
   kick_return kick(expansion L, array<exp_real, NDIM>, kick_stack&, int);

};

struct sort_return {
   tree_ptr check;
   template<class A>
   void serialization(A &&arc, unsigned) {
      assert(false);
   }
};
struct tree {

private:
   array<fixed32, NDIM> pos;
   float radius;
   array<tree_ptr, NCHILD> children;
   std::pair<size_t, size_t> parts;
   multipole *multi;
   static float theta;
   static int8_t rung;
public:
   static particle_set *particles;
   static void set_particle_set(particle_set*);
   inline static fast_future<sort_return> create_child(sort_params&);
   static fast_future<sort_return> cleanup_child();
   static void set_kick_parameters(float theta, int8_t rung);
   static hpx::lcos::local::mutex mtx;
   static std::shared_ptr<kick_workspace_t> get_workspace();
   static void cleanup_workspace(std::shared_ptr<kick_workspace_t>&&);
   static std::stack<std::shared_ptr<kick_workspace_t>> kick_works;
   inline bool is_leaf() const {
      return children[0] == tree_ptr();
   }
   static void cleanup();
   sort_return sort(sort_params = sort_params());
   kick_return kick(expansion L, array<exp_real, NDIM>, kick_stack&, int depth);

   friend class tree_ptr;
};

inline fast_future<array<tree_ptr, NCHILD>> tree_ptr::get_children() const {
   assert(constructed == 1234);
   assert(ptr);
   return fast_future<array<tree_ptr, NCHILD>>(std::move(((tree*) ptr)->children));
}

inline float tree_ptr::get_radius() const {
   assert(constructed == 1234);
   assert(ptr);
   return ((tree*) ptr)->radius;
}

inline array<fixed32, NDIM> tree_ptr::get_pos() const {
   assert(ptr);
   assert(constructed == 1234);
   return ((tree*) ptr)->pos;
}

inline bool tree_ptr::is_leaf() const {
   assert(constructed == 1234);
   assert(ptr);
   return ((tree*) ptr)->children[0] == tree_ptr();
}

