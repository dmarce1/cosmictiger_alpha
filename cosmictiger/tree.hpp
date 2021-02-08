#pragma once

#include <cosmictiger/defs.hpp>
#include <cosmictiger/particle.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/multipole.hpp>
#include <cosmictiger/fast_future.hpp>
#include <cosmictiger/expansion.hpp>
#include <cosmictiger/finite_vector.hpp>
#include <cosmictiger/lockfree_queue.hpp>

#include <queue>
#include <memory>
#include <stack>

#define LEFT 0
#define RIGHT 1
#define WORKSPACE_SIZE 512
#define KICK_GRID_SIZE (128)
#define KICK_BLOCK_SIZE 32
#define KICK_PP_MAX size_t(256)
#define GPU_QUEUE_SIZE (1024*1024)
#define KICK_CUDA_SIZE (1<<15)
#define TREE_PTR_STACK (TREE_MAX_DEPTH*WORKSPACE_SIZE)

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
class kick_params_type;

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
   fast_future<kick_return> kick(kick_params_type*, bool);
#endif
};

struct sort_return {
   tree_ptr check;
   template<class A>
   void serialization(A &&arc, unsigned) {
      assert(false);
   }
};
template<class A, class B>
struct pair {
   A first;
   B second;
};

struct kick_params_stack_type {
   array<tree_ptr, TREE_PTR_STACK> checks;
   array<int, TREE_MAX_DEPTH> counts;
   int count_size;
   int stack_size;
   CUDA_EXPORT inline kick_params_stack_type() {
      THREADID;
      if (tid == 0) {
         count_size = stack_size = 0;
      }CUDA_SYNC();
   }
   CUDA_EXPORT inline tree_ptr* get_top_list() {
      return &checks[stack_size - counts[count_size - 1]];
   }
   CUDA_EXPORT inline
   int& get_top_count() {
      return counts[count_size - 1];
   }
   CUDA_EXPORT inline
   void pop() {
      THREADID;
      if (tid == 0) {
         stack_size -= counts[count_size - 1];
         count_size--;
      }CUDA_SYNC();
   }
   CUDA_EXPORT inline
   void copy_to(tree_ptr *stk, int sz) {
      THREADID;
       BLOCKSIZE;
      for (int i = stack_size + tid; i < stack_size + sz; i += blocksize) {
         checks[i] = stk[i - stack_size];
      }CUDA_SYNC();
      if (tid == 0) {
         stack_size += sz;
         counts[count_size] = sz;
         count_size++;
      }
   }
   CUDA_EXPORT inline
   void copy_top() {
      THREADID;
       BLOCKSIZE;
       for (int i = tid; i < counts[count_size - 1]; i += blocksize) {
         checks[stack_size + i] = checks[stack_size - counts[count_size - 1] + i];
      }CUDA_SYNC();
      if (tid == 0) {
         stack_size += counts[count_size - 1];
         counts[count_size] = counts[count_size - 1];
         count_size++;
      }
   }
   CUDA_EXPORT inline
   void resize_top(int sz) {
      THREADID;
      if (tid == 0) {
         const auto old_sz = counts[count_size - 1];
         counts[count_size - 1] = sz;
         stack_size += sz - old_sz;
      }CUDA_SYNC();
   }
};

struct kick_params_type {
   kick_params_stack_type dstack;
   kick_params_stack_type estack;
   array<tree_ptr, WORKSPACE_SIZE> multi_interactions;
   array<tree_ptr, WORKSPACE_SIZE> part_interactions;
   array<tree_ptr, WORKSPACE_SIZE> next_checks;
   array<tree_ptr, WORKSPACE_SIZE> opened_checks;
   array<expansion, TREE_MAX_DEPTH> L;
   array<array<fixed32, NDIM>, TREE_MAX_DEPTH> Lpos;
   tree_ptr tptr;
   int nmulti;
   int npart;
   int nnext;
   int nopen;
   int depth;
   array<void*,NDIM> pref_ptr;
   size_t pref_size;
   CUDA_EXPORT inline kick_params_type() {
      THREADID;
      if (tid == 0) {
         nmulti = npart = nnext = nopen = depth = 0;
      }CUDA_SYNC();
   }
   friend class tree_ptr;
};

struct kick_params_type;

#ifndef __CUDACC__
struct gpu_kick {
   kick_params_type* params;
   hpx::lcos::local::promise<kick_return> promise;
};
#endif

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
   inline static hpx::future<sort_return> create_child(sort_params&);
   static fast_future<sort_return> cleanup_child();
   static void set_kick_parameters(float theta, int8_t rung);
   static hpx::lcos::local::mutex mtx;
   static hpx::lcos::local::mutex gpu_mtx;
   hpx::future<kick_return> send_kick_to_gpu(kick_params_type *params);
   static void gpu_daemon();
   inline bool is_leaf() const {
      return children[0] == tree_ptr();
   }
   static void cleanup();
   sort_return sort(sort_params = sort_params());
   hpx::future<kick_return> kick(kick_params_type*);
   static std::atomic<bool> daemon_running;
   static std::atomic<bool> shutdown_daemon;
   static lockfree_queue<gpu_kick,GPU_QUEUE_SIZE> gpu_queue;
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

