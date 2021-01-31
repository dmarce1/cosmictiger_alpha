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
#ifdef TEST_STACK
   thread = false;
#endif
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
   const auto& opts = global().opts;
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
   if (part_end - part_begin > opts.bucket_size) {
      std::vector<fast_future<sort_return>> futs(2);
      {
         const auto size = part_end - part_begin;
         auto child_params = params.get_children();
         if (params.key_end - params.key_begin == 1) {
#ifndef TEST_TREE
            const auto &box = params.box;
#endif
            int radix_depth = (int(log(double(size) / opts.bucket_size) / log(2) + TREE_RADIX_CUSHION));
            radix_depth = std::min(std::max(radix_depth, TREE_RADIX_MIN), TREE_RADIX_MAX) + params.depth;
            const auto radix_begin = morton_key(box.begin, radix_depth);
            std::array<fixed64, NDIM> tmp;
            for (int dim = 0; dim < NDIM; dim++) {
               tmp[dim] = box.end[dim] - fixed32::min();
            }
            const auto radix_end = morton_key(tmp, radix_depth) + 1;
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
         for (int ci = 0; ci < NCHILD; ci++) {
            futs[ci] = create_child(child_params[ci]);
         }
      }
      std::array<multipole*, NCHILD> Mc;
      std::array<fixed32*, NCHILD> Xc;
      std::array<float, NCHILD> Rc;
      auto &M = *(self->multi);
      for (int ci = 0; ci < NCHILD; ci++) {
         sort_return rc = futs[ci].get();
         children[ci] = rc.check->client;
         Mc[ci] = rc.check->multi;
         Xc[ci] = rc.check->pos.data();
         self->children[ci] = rc.check;
      }
      self->leaf = false;
      std::array<double, NDIM> com = { 0, 0, 0 };
      const auto &MR = *Mc[RIGHT];
      const auto &ML = *Mc[LEFT];
      M() = ML() + MR();
      double rleft = 0.0;
      double rright = 0.0;
      for (int dim = 0; dim < NDIM; dim++) {
         com[dim] = (ML() * Xc[LEFT][dim].to_double() + MR() * Xc[RIGHT][dim].to_double()) / (ML() + MR());
         self->pos[dim] = com[dim];
         rleft += (Xc[LEFT][dim].to_double() - com[dim]) * (Xc[LEFT][dim].to_double() - com[dim]);
         rright += (Xc[RIGHT][dim].to_double() - com[dim]) * (Xc[RIGHT][dim].to_double() - com[dim]);
      }
      std::array<double, NDIM> xl, xr;
      for (int dim = 0; dim < NDIM; dim++) {
         xl[dim] = Xc[LEFT][dim].to_double() - com[dim];
         xr[dim] = Xc[RIGHT][dim].to_double() - com[dim];
      }
      M = (ML >> xl) + (MR >> xr);
      rleft = std::sqrt(rleft) + self->children[LEFT]->radius;
      rright = std::sqrt(rright) + self->children[RIGHT]->radius;
      self->radius = std::max(rleft, rright);
      //     printf("x      = %e\n", self->pos[0].to_float());
      //    printf("y      = %e\n", self->pos[1].to_float());
      //   printf("z      = %e\n", self->pos[2].to_float());
      //  printf("radius = %e\n", self->radius);
   } else {
      std::array<double, NDIM> com = { 0, 0, 0 };
      for (auto i = part_begin; i < part_end; i++) {
         for (int dim = 0; dim < NDIM; dim++) {
            com[dim] += particles->pos(dim, i).to_double();
         }
      }
      for (int dim = 0; dim < NDIM; dim++) {
         com[dim] /= (part_end - part_begin);
         self->pos[dim] = com[dim];

      }
      auto &M = *(self->multi);
      M = 0.0;
      self->radius = 0.0;
      for (auto i = part_begin; i < part_end; i++) {
         double this_radius = 0.0;
         M() += 1.0;
         for (int n = 0; n < NDIM; n++) {
            const auto xn = particles->pos(n, i).to_double() - com[n];
            this_radius += xn * xn;
            for (int m = n; m < NDIM; m++) {
               const auto xm = particles->pos(m, i).to_double() - com[m];
               M(n, m) += xn * xm;
               for (int l = m; l > NDIM; l++) {
                  const auto xl = particles->pos(l, i).to_double() - com[l];
                  M(n, m, l) -= xn * xm;
               }
            }
         }
         this_radius = std::sqrt(this_radius);
         self->radius = std::max(self->radius, (float) (this_radius));
      }
      self->parts.first = part_begin;
      self->parts.second = part_end;
      self->leaf = true;
   }
   self->client.rank = hpx_rank();
   self->client.ptr = (uint64_t) this;
   sort_return rc;
   rc.check = self;
   //  if (params.depth == 0) {
   // }
   return rc;
}
