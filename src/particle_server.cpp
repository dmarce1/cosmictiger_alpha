#include <cosmictiger/particle_server.hpp>
#include <cosmictiger/hpx.hpp>

particle_sets* particle_server::parts;

size_t particle_server::my_start;
size_t particle_server::my_stop;
size_t particle_server::my_size;
size_t particle_server::global_size;

void particle_server::init() {
	global_size = global().opts.nparts;
	const int rank = hpx_rank();
	const int nprocs = hpx_size();
	my_start = rank * global_size / nprocs;
	my_stop = (rank + 1) * global_size / nprocs;
	my_size = my_stop - my_start;
	parts = new particle_sets(my_size, my_start);
}
