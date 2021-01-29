#include <cosmictiger/global.hpp>
#include <cosmictiger/tree.hpp>

#include <cmath>

particle_set* tree::particles;

void tree::set_particle_set(particle_set* parts) {
   particles = parts;
}

tree::tree() {
}

/*
 range box;
 int depth;
 int radix_depth;
 std::shared_ptr<std::vector<size_t>> bounds;
 size_t key_begin;
 size_t key_end;
 */

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
   const auto &box = params->box;
   const size_t size = part_end - part_begin;
   if (size > opts.bucket_size) {
      auto child_params = params->get_children();
      if (params->key_end - params->key_end <= 1) {
         const int radix_depth = (int(log(double(size) / opts.bucket_size) / log(8)) + 1) * NDIM + params->radix_depth;
         printf("Sorting to depth %i\n", radix_depth);
         const auto key_begin = morton_key(box.begin, radix_depth);
         const auto key_end = morton_key(box.end, radix_depth) + 1;
         auto bounds = particles->local_sort(part_begin, part_end, radix_depth, key_begin, key_end);
         auto bndptr = std::make_shared<decltype(bounds)>(std::move(bounds));
         for (int ci = 0; ci < NCHILD; ci++) {
            child_params[ci].key_begin = 0;
            child_params[ci].key_end = key_end - key_begin;
            child_params[ci].bounds = bndptr;
            child_params[ci].radix_depth = radix_depth;
         }
      }


   }
   return rc;
}
