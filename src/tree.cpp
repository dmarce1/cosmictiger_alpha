#include <cosmictiger/global.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/memory.hpp>
#include <cosmictiger/particle.hpp>
#include <cosmictiger/thread_control.hpp>

#define LEFT 0
#define RIGHT 1

HPX_PLAIN_ACTION(tree::create, tree_create_action);

tree::tree() :
      tree(nullptr) {

}

tree::tree(std::shared_ptr<tree::sort_vars> vars) {
   const auto &opts = global().opts;
   if (vars == nullptr) {
      vars = std::make_shared<sort_vars>();
      CHECK_POINTER(vars);
      for (int dim = 0; dim < NDIM; dim++) {
         vars->box.begin[dim] = fixed32::min();
         vars->box.end[dim] = fixed32::max();
      }
      vars->depth = 0;
      vars->bounds.push_back(0);
      vars->bounds.push_back(opts.nparts);
   }
   const size_t nbins = vars->bounds.size();
   parts_begin = vars->bounds[0];
   parts_end = vars->bounds[nbins - 1];
 //  printf( "Forming tree at depth %i %li %li\n", vars->depth, parts_begin, parts_end);
#ifdef TEST_TREE
   const auto set = particle_set::local_particle_set();
   bool error = false;
   for (size_t i = parts_begin; i < parts_end; i++) {
      const auto x = set.get_x(i);
      if (!vars->rng.contains(x)) {
         printf("particle %li out of range at depth = %i\n", i, vars->depth);
         printf("Range:\n");
         for (int dim = 0; dim < NDIM; dim++) {
            printf("\t%e %e\n", vars->rng.begin[dim].to_float(), vars->rng.end[dim].to_float());
         }
         printf("Particle:\n\t");
         for (int dim = 0; dim < NDIM; dim++) {
            printf("%e ", x[dim].to_float());
         }
         printf("\n");
         error = true;
      }
      if (error) {
         abort();
      }
   }
#endif
   if (parts_end - parts_begin > opts.bucket_size) {
      std::array<std::shared_ptr<sort_vars>, NCHILD> child_vars;
      std::array<hpx::future<id_type>, NCHILD> futs;
      auto &bounds = vars->bounds;
      const int sort_dim = vars->depth % NDIM;
      if (vars->bounds.size() <= 2) {
         bounds = particle_set::radix_sort(parts_begin, parts_end, vars->box, sort_dim, RADIX_DEPTH);
      }
      const fixed32 middle_x = (fixed64(vars->box.begin[sort_dim]) + fixed64(vars->box.end[sort_dim])) / fixed64(2.0f);
      const size_t N = bounds.size() / 2;
      for (int i = 0; i < NCHILD; i++) {
         child_vars[i] = std::make_shared<sort_vars>();
         auto &cbounds = child_vars[i]->bounds;
         cbounds.insert(cbounds.end(), bounds.begin() + i * N, bounds.begin() + (i + 1) * N + 1);
         child_vars[i]->depth = vars->depth + 1;
         child_vars[i]->box = vars->box;
         if (i == LEFT) {
            child_vars[LEFT]->box.end[sort_dim] = middle_x;
         } else if (i == RIGHT) {
            child_vars[RIGHT]->box.begin[sort_dim] = middle_x;
         }
      }
      for (int i = 0; i < NCHILD; i++) {
         futs[i] = create(particle_set::index_to_rank(child_vars[i]->bounds[0]), child_vars[i]);
      }
      for (int i = 0; i < NCHILD; i++) {
         children[i] = futs[i].get();
      }
   }
}

hpx::future<tree::id_type> tree::create(int rank, std::shared_ptr<tree::sort_vars> vars) {
   static const int myrank = hpx_rank();
   hpx::future<id_type> fut;
   if (rank == myrank) {
//      const size_t nparts = vars->end - vars->begin;
//      const int nthreads =
//            nparts < NPARTS_FULLSYSTEM_SEARCH ? 1 : hpx::thread::hardware_concurrency() * OVERSUBSCRIPTION;
//      thread_control thread(nthreads, vars->depth);
      const auto func = [=](std::shared_ptr<tree::sort_vars> vars) {
         id_type id;
         tree *ptr = new tree(vars);
         CHECK_POINTER(ptr);
         id.ptr = (uint64_t) ptr;
         id.rank = myrank;
//         if (vars->forked) {
//            vars->thread.release();
//         }
         return id;
      };
//      if (thread.try_acquire()) {
//         vars->forked = true;
//         vars->thread = std::move(thread);
//         fut = hpx::async(hpx::launch::async, func, std::move(vars));
//      } else {
//         vars->forked = false;
//         vars->thread = std::move(thread);
         fut = hpx::make_ready_future(func(std::move(vars)));
//      }
   } else {
      fut = hpx::async < tree_create_action > (hpx_localities()[rank], rank, vars);
   }
   return fut;
}

