#include <cosmictiger/particle.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/memory.hpp>


void particle_set::create() {
   const auto nranks = hpx_localities().size();
   const auto &myrank = hpx_rank();
   const size_t &nparts = global().opts.nparts;
   const size_t mystart = myrank * nparts / nranks;
   const size_t mystop = (myrank + 1) * nparts / nranks;
   parts.offset = -mystart;
   parts.size = mystop - mystart;
   parts.virtual_ = false;
   for (int dim = 0; dim < NDIM; dim++) {
      CUDA_MALLOC(parts.x[dim], nparts);
      CUDA_MALLOC(parts.v[dim], nparts);
   }
   CUDA_MALLOC(parts.rung, nparts);
   parts.size = nparts;
}
