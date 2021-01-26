#include <cosmictiger/global.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/memory.hpp>
#include <cosmictiger/particle_sort.hpp>

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
         vars->rng.begin[dim] = fixed32::min();
         vars->rng.end[dim] = fixed32::max();
      }
      vars->depth = 0;
      vars->begin = 0;
      vars->end = opts.nparts;
   }
   parts_begin = vars->begin;
   parts_end = vars->end;

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
      {
         const int sort_dim = vars->depth % NDIM;
         const fixed32 sort_pos = (fixed64(vars->rng.begin[sort_dim]) + fixed64(vars->rng.end[sort_dim]))
               / fixed64(2.0f);
   //      printf("Sorting %li %li\n", parts_begin, parts_end);
         const size_t mid = particle_set::sort(parts_begin, parts_end, sort_dim, sort_pos);
  //       printf("Done sorting %li %li %li\n", parts_begin, mid, parts_end);
         for (int i = 0; i < NCHILD; i++) {
            child_vars[i] = std::make_shared<sort_vars>();
            child_vars[i]->begin = i == LEFT ? parts_begin : mid;
            child_vars[i]->end = i == LEFT ? mid : parts_end;
            child_vars[i]->depth = vars->depth + 1;
            child_vars[i]->rng = vars->rng;
            if (i == LEFT) {
               child_vars[LEFT]->rng.end[sort_dim] = sort_pos;
            } else if (i == RIGHT) {
               child_vars[RIGHT]->rng.begin[sort_dim] = sort_pos;
            }
         }
         for (int i = 0; i < NCHILD; i++) {
            futs[i] = create(particle_set::index_to_rank(child_vars[i]->begin), child_vars[i]);
         }
         for (int i = 0; i < NCHILD; i++) {
            children[i] = futs[i].get();
         }
      }
   }
}

hpx::future<tree::id_type> tree::create(int rank, std::shared_ptr<tree::sort_vars> vars) {
   static const int myrank = hpx_rank();
   hpx::future<id_type> fut;
   if (rank == myrank) {
      id_type id;
      tree *ptr = new tree(vars);
      id.ptr = (uint64_t) ptr;
      id.rank = myrank;
      fut = hpx::make_ready_future(id);
   } else {
      fut = hpx::async < tree_create_action > (hpx_localities()[rank], rank, vars);
   }
   return fut;
}

