#include <cosmictiger/defs.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/particle.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/tests.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/ewald_indices.hpp>


void yield() {
	hpx::this_thread::yield();
}

int hpx_main(int argc, char *argv[]) {
	options opts;
	//  printf( "%li\n", sizeof(std::shared_ptr<int>));
	//  printf( "%li\n", sizeof(sort_params));
	printf("Size of cuda_kick_shmem is %li\n", sizeof(cuda_kick_shmem));
	printf("Size of cuda_ewald_shmem is %li\n", sizeof(cuda_ewald_shmem));
	if (process_options(argc, argv, opts)) {
		hpx_init();
		const auto cuda = cuda_init();
		printf("Initializing ewald\n");
		ewald_const::init();
		printf("Done initializing ewald\n");
		global_init(opts, cuda);

		if (opts.test != "") {
			test_run(opts.test);
		} else {

		}
	}
	tree::cleanup();
	return hpx::finalize();
}

#ifdef USE_HPX
int main(int argc, char *argv[]) {
	std::vector < std::string > cfg = {"hpx.commandline.allow_unknown=1"};
	cfg.push_back("hpx.stacks.small_size=2097152");
	hpx::init(argc, argv, cfg);
}
#endif
