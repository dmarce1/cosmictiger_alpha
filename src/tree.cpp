#include <cosmictiger/global.hpp>
#include <cosmictiger/tree.hpp>

#include <cmath>

particle_set *tree::particles;

void tree::set_particle_set(particle_set *parts) {
   particles = parts;
}

tree::tree() {
}

hpx::future<sort_return> tree::create_child(std::shared_ptr<sort_params> params) {
   tree_client id;
   id.rank = 0;
   id.ptr = (uintptr_t) new tree;
   CHECK_POINTER(id.ptr);
   sort_return rc = ((tree*) (id.ptr))->sort(params);
   rc.client = id;
   return hpx::make_ready_future(rc);
}

sort_return tree::sort(std::shared_ptr<sort_params> params) {
   const auto opts = global().opts;
   sort_return rc;

   if (params == nullptr) {
      params = std::make_shared<sort_params>();
      params->set_root();
   }

   const auto bnds = params->get_bounds();
   part_begin = bnds.first;
   part_end = bnds.second;
   depth = params->depth;
   if (depth == TREE_MAX_DEPTH) {
      printf("Exceeded maximum tree depth\n");

      abort();
   }


//   printf( "Creating tree node at %i %li %li %li %li\n", depth, part_begin, part_end, params->key_begin, params->key_end);
   const auto &box = params->box;
   const size_t size = part_end - part_begin;

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
      abort();
   }
#endif
   if (size > opts.bucket_size) {
      auto child_params = params->get_children();
    //  printf( "%li %li %li\n",depth, params->key_end, params->key_begin );
      if (params->key_end - params->key_begin == 1) {
         int radix_depth = std::max((int(log(double(size) / opts.bucket_size) / log(8))),1) * NDIM + params->radix_depth;
         radix_depth = std::min(TREE_MAX_RADIX, radix_depth);
         printf("------->Sorting to depth %i from level %i\n", radix_depth, depth);
         const auto key_begin = morton_key(box.begin, radix_depth);
         const auto key_end = morton_key(box.end, radix_depth) + 1;
         auto bounds = particles->local_sort(part_begin, part_end, radix_depth, key_begin, key_end);
         printf("Done sorting\n");
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
         futs[ci] = create_child(std::make_shared < sort_params > (child_params[ci]));
      }
      for (int ci = 0; ci < NCHILD; ci++) {
         sort_return rc = futs[ci].get();
         children[ci] = rc.client;
      }

   }
   return rc;
}
