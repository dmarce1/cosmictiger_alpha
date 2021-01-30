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

hpx::future<sort_return> tree::create_child(sort_params* params) {
   tree_client id;
   id.rank = 0;
   id.ptr = (uintptr_t) params->allocs->tree_alloc->allocate();
   CHECK_POINTER(id.ptr);
#ifdef TREE_SORT_MULTITHREAD
   thread_control thread(1);
   const auto nparts = (*params->bounds)[params->key_end] - (*params->bounds)[params->key_begin];
   if( nparts > TREE_MIN_PARTS2THREAD &&  thread.try_acquire()) {
      params->allocs = std::make_shared<tree_alloc>();
      params->allocs->multi_alloc = std::make_shared<managed_allocator<multipole>>();
      params->allocs->check_alloc = std::make_shared<managed_allocator<check_item>>();
      params->allocs->tree_alloc = std::make_shared<managed_allocator<tree>>();
      params->allocs->params_alloc = std::make_shared<managed_allocator<sort_params>>();

      return hpx::async([id,params](thread_control&& thread) {
         auto rc = ((tree*) (id.ptr))->sort(params);
         thread.release();
         return rc;
      }, std::move(thread));
   } else {
      return hpx::make_ready_future( ((tree*) (id.ptr))->sort(params));
   }
#else
   return hpx::make_ready_future( ((tree*) (id.ptr))->sort(params));
#endif
}

sort_return tree::sort(sort_params* params) {
   const auto opts = global().opts;
   sort_return rc;

   if (params == nullptr) {
      auto alloc = std::make_shared<managed_allocator<sort_params>>();
      params = alloc->allocate();
      params->set_root();
      params->allocs->params_alloc = alloc;
   }
   allocs = params->allocs;
   self = allocs->check_alloc->allocate();
   const auto bnds = params->get_bounds();
   part_begin = bnds.first;
   part_end = bnds.second;
   if (params->depth == TREE_MAX_DEPTH) {
      printf("Exceeded maximum tree depth\n");

      abort();
   }

 //   printf( "Creating tree node at %i %li %li %li %li\n", depth, part_begin, part_end, params->key_begin, params->key_end);
   const auto &box = params->box;
   const size_t size = part_end - part_begin;

   self = allocs->check_alloc->allocate();
   self->multi = allocs->multi_alloc->allocate();

#ifdef TEST_TREE
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
   if (size > opts.bucket_size) {
      auto child_params = params->get_children();
    //  printf( "%li %li %li\n",depth, params->key_end, params->key_begin );
      if (params->key_end - params->key_begin == 1) {
         int radix_depth = (int(log(double(size) / opts.bucket_size) / log(8)))+TREE_RADIX_CUSHION;
         radix_depth = std::min(std::max(radix_depth,1) * NDIM,TREE_RADIX_MAX) + params->radix_depth;
   //        printf("------->Sorting to depth %i from level %i\n", radix_depth, depth);
         const auto key_begin = morton_key(box.begin, radix_depth);
         std::array<fixed64, NDIM> tmp;
         for( int dim = 0; dim < NDIM; dim++) {
            tmp[dim] = box.end[dim] - fixed32::min();
         }
         const auto key_end = morton_key(tmp, radix_depth) + 1;

      //   printf( "Key begin = %llx %llx\n", key_begin, key_end);
         for( int dim = 0; dim < NDIM; dim++) {
         //   printf( "%e %e\n", box.begin[dim].to_float(), box.end[dim].to_float());
          //  printf( "%lx %lx\n", box.begin[dim].get_integer(), box.end[dim].get_integer());

         }
         timer tm;
         tm.start();
//         printf( "***************** %li %li ****************\n", part_begin,part_end);
         auto bounds = particles->local_sort(part_begin, part_end, radix_depth, key_begin, key_end);
//         printf( "bounds.size %li\n", bounds.size());
//         for( int i = 0; i < bounds.size(); i++) {
//            printf( "%lx\n", bounds[i]);
//         }
         assert(bounds[0] >= part_begin);
         assert(bounds[bounds.size()-1] <= part_end);
         tm.stop();
    //     printf("Done sorting %e\n", tm.read());
         auto bndptr = std::make_shared<decltype(bounds)>(std::move(bounds));
         for (int ci = 0; ci < NCHILD; ci++) {
            child_params[ci].bounds = bndptr;
            child_params[ci].radix_depth = radix_depth;
         }
         child_params[LEFT].key_begin = 0;
         child_params[LEFT].key_end = child_params[RIGHT].key_begin = (key_end - key_begin) / 2;
         child_params[RIGHT].key_end = (key_end - key_begin);
      }
   //   printf("Creating children depth = %li\n", params->depth);
      std::array<hpx::future<sort_return>, NCHILD> futs;
      for (int ci = 0; ci < NCHILD; ci++) {
         auto* ptr = params->allocs->params_alloc->allocate();
         *ptr = child_params[ci];
         futs[ci] = create_child(ptr);
      }
      for (int ci = 0; ci < NCHILD; ci++) {
         sort_return rc = futs[ci].get();
         children[ci] = rc.check->client;
         self->children[ci]= rc.check;
         self->leaf = false;
      }
   } else {
      self->parts.first = part_begin;
      self->parts.second = part_end;
      self->leaf = false;
   }
   self->client.rank = hpx_rank();
   self->client.ptr = (uint64_t) this;
   rc.check = self;
   return rc;
}
