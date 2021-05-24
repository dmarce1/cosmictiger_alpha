#include <cosmictiger/defs.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/initial.hpp>
#include <cosmictiger/particle.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/tests.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/ewald_indices.hpp>
#include <cosmictiger/driver.hpp>
#include <cosmictiger/tree_database.hpp>
#include <cosmictiger/zero_order.hpp>

#include <gperftools/malloc_extension.h>

void yield() {
	hpx::this_thread::yield();
}

int hpx_main(int argc, char *argv[]) {
	options opts;
	//  PRINT( "%li\n", sizeof(std::shared_ptr<int>));
	//  PRINT( "%li\n", sizeof(sort_params));
	PRINT("Size of cuda_kick_shmem is %li\n", sizeof(cuda_kick_shmem));

	if (process_options(argc, argv, opts)) {

		hpx_init();
		const auto cuda = cuda_init();
		PRINT("Initializing ewald\n");
		ewald_const::init();
		PRINT("Done initializing ewald\n");
		global_init(opts, cuda);
//		tree_data_initialize();
		if (opts.test != "") {
			test_run(opts.test);
		} else {
			drive_cosmos();
		}
	}
	tree::cleanup();
	return hpx::finalize();
}

#ifdef USE_HPX

#include <oneapi/tbb/scalable_allocator.h>

int main(int argc, char *argv[]) {
	std::vector<std::string> cfg = { "hpx.commandline.allow_unknown=1" };
	cfg.push_back("hpx.stacks.small_size=2097152");
	hpx::init(argc, argv, cfg);
}

#endif
