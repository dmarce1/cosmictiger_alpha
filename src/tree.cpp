#include <cosmictiger/global.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/thread_control.hpp>

#include <cmath>

particle_set *tree::particles;

void tree::set_particle_set(particle_set *parts) {
   particles = parts;
}

tree::tree() {
}

inline fast_future<sort_return> tree::create_child(sort_params &params) {
   static std::atomic<int> threads_used(0);
   tree_client id;
   id.rank = 0;
   id.ptr = (uintptr_t) params.allocs->tree_alloc.allocate();
   CHECK_POINTER(id.ptr);
   const auto nparts = (*params.bounds)[params.key_end] - (*params.bounds)[params.key_begin];
   bool thread = false;
   if (nparts > TREE_MIN_PARTS2THREAD) {
      if (++threads_used <= hpx::thread::hardware_concurrency()) {
         thread = true;
      } else {
         threads_used--;
      }
   }
   if (!thread) {
      return fast_future<sort_return>(((tree*) (id.ptr))->sort(params));
   } else {
      params.allocs = std::make_shared<tree_alloc>();
      return hpx::async([id, params]() {
         auto rc = ((tree*) (id.ptr))->sort(params);
         threads_used--;
         return rc;
      });
   }
}

sort_return tree::sort(sort_params params) {
   const auto opts = global().opts;
   if (params.iamroot()) {
      params.set_root();
   }
   self = params.allocs->check_alloc.allocate();
   {
      const auto bnds = params.get_bounds();
      part_begin = bnds.first;
      part_end = bnds.second;
   }
   if (params.depth == TREE_MAX_DEPTH) {
      printf("Exceeded maximum tree depth\n");
      abort();
   }

   self = params.allocs->check_alloc.allocate();
   self->multi = params.allocs->multi_alloc.allocate();

#ifdef TEST_TREE
   const auto &box = params.box;
   bool failed = false;
   for( size_t i = part_begin; i < part_end; i++) {
      particle p = particles->part(i);
      if( !box.contains(p.x)) {
         printf( "Particle out of range !\n");
         printf( "Box\n");
         for( int dim = 0; dim < NDIM; dim++) {
            printf( "%e %e |", box.begin[dim].to_float(), box.end[dim].to_float());
         }
         printf( "\n");
         printf( "Particle\n");
         for( int dim = 0; dim < NDIM; dim++) {
            printf( "%e ", p.x[dim].to_float());
         }
         printf( "\n");
  //       abort();
         failed = true;
      }
   }
   if( failed ) {
     // abort();
   }
#endif
#ifdef TEST_STACK
   {
      uint8_t dummy;
      printf("Stack usaged = %li Depth = %li \n", &dummy - params.stack_ptr, params.depth);
   }
#endif
   if ( part_end - part_begin > opts.bucket_size) {
      const auto size =  part_end - part_begin;
      auto child_params = params.get_children();
      if (params.key_end - params.key_begin == 1) {
         int radix_depth = (int(log(double(size) / opts.bucket_size) / log(2) + TREE_RADIX_CUSHION));
         radix_depth = std::min(std::max(radix_depth, TREE_RADIX_MIN), TREE_RADIX_MAX) + params.depth;
         const auto radix_begin = params.radix_begin >> (TREE_RADIX_MAX - radix_depth);
         const auto radix_end = params.radix_end >> (TREE_RADIX_MAX - radix_depth);
         //      printf( "Radix depth = %li\n", radix_depth);
         auto bounds = particles->local_sort(part_begin, part_end, radix_depth, radix_begin, radix_end);
         assert(bounds[0] >= part_begin);
         assert(bounds[bounds.size() - 1] <= part_end);
         auto bndptr = std::make_shared<decltype(bounds)>(std::move(bounds));
         for (int ci = 0; ci < NCHILD; ci++) {
            child_params[ci].bounds = bndptr;
         }
         child_params[LEFT].key_begin = 0;
         child_params[LEFT].key_end = child_params[RIGHT].key_begin = (radix_end - radix_begin) / 2;
         child_params[RIGHT].key_end = (radix_end - radix_begin);
      }
      std::array<fast_future<sort_return>, NCHILD> futs;
      for (int ci = 0; ci < NCHILD; ci++) {
         futs[ci] = create_child(child_params[ci]);
      }
      for (int ci = 0; ci < NCHILD; ci++) {
         sort_return rc = futs[ci].get();
         children[ci] = rc.check->client;
         self->children[ci] = rc.check;
         self->leaf = false;
      }
   } else {
      self->parts.first = part_begin;
      self->parts.second = part_end;
      self->leaf = false;
   }
   self->client.rank = hpx_rank();
   self->client.ptr = (uint64_t) this;
   sort_return rc;
   rc.check = self;
   return rc;
}
