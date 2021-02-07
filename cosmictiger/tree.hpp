#pragma once

#include <cosmictiger/defs.hpp>
#include <cosmictiger/particle.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/multipole.hpp>
#include <cosmictiger/fast_future.hpp>
#include <cosmictiger/expansion.hpp>
#include <cosmictiger/finite_vector.hpp>

#include <queue>
#include <memory>
#include <stack>

#define LEFT 0
#define RIGHT 1
#define WORKSPACE_SIZE 512
#define KICK_GRID_SIZE 128
#define KICK_BLOCK_SIZE 32
#define N_CUDA_WORKSPACE 8

#define EWALD_MIN_DIST2 (0.25f * 0.25f)

class tree;
struct sort_params;
struct tree_ptr;

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

struct kick_return {
   int8_t rung;
};

class tree_ptr;

using checks_type = finite_vector<tree_ptr,WORKSPACE_SIZE>;

struct kick_stack;

struct tree_ptr {
   uintptr_t ptr;
   int rank;
   int8_t opened;
#ifndef NDEBUG
   int constructed;
#endif
   CUDA_EXPORT inline tree_ptr() {
      rank = -1;
      ptr = 0;
      opened = false;
#ifndef NDEBUG
      constructed = 1234;
#endif
   }
   CUDA_EXPORT inline tree_ptr(tree_ptr &&other) {
      rank = other.rank;
      ptr = other.ptr;
      opened = other.opened;
#ifndef NDEBUG
      constructed = 1234;
#endif
   }
   CUDA_EXPORT inline tree_ptr(const tree_ptr &other) {
      rank = other.rank;
      ptr = other.ptr;
      opened = other.opened;
#ifndef NDEBUG
      constructed = 1234;
#endif
   }
   CUDA_EXPORT inline tree_ptr& operator=(const tree_ptr &other) {
      assert(constructed == 1234);
      ptr = other.ptr;
      rank = other.rank;
      opened = other.opened;
      return *this;
   }
   CUDA_EXPORT
   inline tree_ptr& operator=(tree_ptr &&other) {
      assert(constructed == 1234);
      ptr = other.ptr;
      rank = other.rank;
      opened = other.opened;
      return *this;
   }
   CUDA_EXPORT
   inline bool operator==(const tree_ptr &other) const {
      assert(constructed == 1234);
      return rank == other.rank && ptr == other.ptr && opened == other.opened;
   }
   template<class A>
   void serialization(A &&arc, unsigned) {
      arc & ptr;
      arc & rank;
      arc & opened;
   }
   CUDA_EXPORT

   inline operator tree*() {
      assert(constructed == 1234);
      assert(ptr);
      return (tree*) (ptr);
   }
   CUDA_EXPORT
   inline operator const tree*() const {
      assert(constructed == 1234);
      assert(ptr);
      return (const tree*) (ptr);
   }
#ifndef __CUDACC__
   fast_future<array<tree_ptr, NCHILD>> get_children() const;
#else
   CUDA_EXPORT
 array<tree_ptr,NCHILD> get_children() const ;
#endif
   CUDA_EXPORT
   float get_radius() const;
   CUDA_EXPORT
   array<fixed32, NDIM> get_pos() const;
   CUDA_EXPORT
   bool is_leaf() const;
#ifndef __CUDACC__
   fast_future<kick_return> kick(kick_stack&, int, bool);
#endif
};

struct kick_workspace_t {
   checks_type multi_interactions;
   checks_type part_interactions;
   checks_type next_checks;
   checks_type opened_checks;
};

#define WORKSPACE_STACK (64*1024)

struct check_stack {
   finite_vector<tree_ptr, WORKSPACE_STACK> stack;
   finite_vector<int, WORKSPACE_STACK> counts;
   CUDA_EXPORT tree_ptr* top_list() {
      return &stack[stack.size() - counts.top()];
   }
   CUDA_EXPORT int top_count() const {
      return counts.top();
   }
   CUDA_EXPORT void set_top_size(size_t sz) {
      stack.resize(stack.size() - counts.top() + sz);
      counts.top() = sz;
   }
   CUDA_EXPORT void pop() {
      stack.resize(stack.size() - counts.top());
      counts.pop_back();
   }
   CUDA_EXPORT void copy_top(check_stack &other) {
      THREADID;
      BLOCKSIZE;
      if (&other != this) {
         const size_t offset = other.stack.size();
         other.stack.resize(offset + counts.top());
         for (int i = tid; i < counts.top(); i += blocksize) {
            other.stack[offset + i] = top_list()[i];
         }
         other.counts.push_back(counts.top());
      } else {
         const size_t offset = stack.size();
         stack.resize(offset + counts.top());
         for (int i = tid; i < counts.top(); i += blocksize) {
            stack[offset + i] = stack[offset + i - counts.top()];
         }
         counts.push_back(counts.top());
      }
   }
   template<size_t N>
   CUDA_EXPORT void push(const finite_vector<tree_ptr, N> &list) {
      THREADID;
      BLOCKSIZE;
      size_t offset = stack.size();
      stack.resize(offset + list.size());
      for (int i = tid; i < list.size(); i += blocksize) {
         stack[offset + i] = list[i];
      }
      counts.push_back(list.size());
   }

};

struct kick_stack {
   check_stack dchecks;
   check_stack echecks;
   finite_vector<expansion, TREE_MAX_DEPTH> L;
   finite_vector<array<fixed32, NDIM>, TREE_MAX_DEPTH> Lpos;
   kick_stack(kick_stack&&) = default;
   kick_stack& operator=(kick_stack&&) = default;
   kick_stack() {
      L.resize(TREE_MAX_DEPTH);
      Lpos.resize(TREE_MAX_DEPTH);
   }
};

struct sort_return {
   tree_ptr check;
   template<class A>
   void serialization(A &&arc, unsigned) {
      assert(false);
   }
};

#ifndef __CUDACC__
struct gpu_kick {
   tree_ptr tree;
   kick_workspace_t space;
   kick_stack stack;
   int depth;
   hpx::lcos::local::promise<kick_return> promise;
   gpu_kick(gpu_kick&&) = default;
   gpu_kick() = default;

};
#endif

template<class A, class B>
struct pair {
   A first;
   B second;
};

struct cuda_workspace_t {
   kick_stack *stacks;
   tree_ptr *roots;
   kick_workspace_t *workspace;
   kick_return *rc;
   int *depths;
   int8_t *ptr;
   size_t grid_size;
   cuda_workspace_t(size_t grid_size_) {
      grid_size = grid_size_;
      const size_t sz = grid_size
            * (sizeof(kick_stack) + sizeof(tree_ptr) + sizeof(int) + sizeof(kick_workspace_t) + sizeof(kick_return));
      CUDA_MALLOC(ptr, sz);
      stacks = (kick_stack*) ptr;
      roots = (tree_ptr*) (stacks + grid_size);
      workspace = (kick_workspace_t*) (roots + grid_size);
      depths = (int*) (workspace + grid_size);
      rc = (kick_return*) (workspace + grid_size);
      for (int i = 0; i < grid_size; i++) {
         new (stacks + i) kick_stack();
         new (workspace + i) kick_workspace_t();
         new (rc + i) kick_return();
         new (roots + i) tree_ptr();
      }
   }
   ~cuda_workspace_t() {
      for (int i = 0; i < grid_size; i++) {
         stacks[i].kick_stack::~kick_stack();
         roots[i].tree_ptr::~tree_ptr();
         workspace[i].kick_workspace_t::~kick_workspace_t();
         rc[i].kick_return::~kick_return();
      }
      CUDA_FREE(ptr);
   }
};

struct tree {

#ifndef __CUDACC__
private:
#endif
   array<fixed32, NDIM> pos;
   float radius;
   array<tree_ptr, NCHILD> children;
   pair<size_t, size_t> parts;
   multipole *multi;
   static float theta;
   static int8_t rung;
public:
   static particle_set *particles;
   static void set_cuda_particle_set(particle_set*);
   static void cuda_set_kick_params(particle_set *p, float theta_, int rung_);
#ifndef __CUDACC__
   static void set_particle_set(particle_set*);
   static std::shared_ptr<kick_workspace_t> get_workspace();
   static void cleanup_workspace(std::shared_ptr<kick_workspace_t>&&);
   inline static hpx::future<sort_return> create_child(sort_params&);
   static fast_future<sort_return> cleanup_child();
   static void set_kick_parameters(float theta, int8_t rung);
   static hpx::lcos::local::mutex mtx;
   static hpx::lcos::local::mutex gpu_mtx;
   static std::stack<std::shared_ptr<kick_workspace_t>> kick_works;
   hpx::future<kick_return> send_kick_to_gpu(kick_stack &stack, int depth);
   static void gpu_daemon();
   inline bool is_leaf() const {
      return children[0] == tree_ptr();
   }
   static void cleanup();
   sort_return sort(sort_params = sort_params());
   hpx::future<kick_return> kick(kick_stack&, int depth);
   static bool daemon_running;
   static bool shutdown_daemon;
   static std::queue<gpu_kick> gpu_queue;
#endif
   friend class tree_ptr;
};

#ifndef __CUDACC__
inline fast_future<array<tree_ptr, NCHILD>> tree_ptr::get_children() const {
   assert(constructed == 1234);
   assert(ptr);
   return fast_future<array<tree_ptr, NCHILD>>(((tree*) ptr)->children);
}
#else
CUDA_EXPORT
inline array<tree_ptr, NCHILD> tree_ptr::get_children() const {
   assert(constructed == 1234);
   assert(ptr);
   return (((tree*) ptr)->children);
}

#endif

CUDA_EXPORT
inline float tree_ptr::get_radius() const {
   assert(constructed == 1234);
   assert(ptr);
   return ((tree*) ptr)->radius;
}

CUDA_EXPORT
inline array<fixed32, NDIM> tree_ptr::get_pos() const {
   assert(ptr);
   assert(constructed == 1234);
   return ((tree*) ptr)->pos;
}

CUDA_EXPORT
inline bool tree_ptr::is_leaf() const {
   assert(constructed == 1234);
   assert(ptr);
   return ((tree*) ptr)->children[0] == tree_ptr();
}

