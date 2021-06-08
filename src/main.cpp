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
#include <cosmictiger/tensor.hpp>
void yield() {
	hpx::this_thread::yield();
}

int hpx_main(int argc, char *argv[]) {
	options opts;
	//  PRINT( "%li\n", sizeof(std::shared_ptr<int>));
	//  PRINT( "%li\n", sizeof(sort_params));

	constexpr int N = 4;
	tensor_sym<float, N> test;
	for (int i = 0; i < N * (N + 1) * (N + 2) / 6; i++) {
		test[i] = 2.0 * float(rand()) / float(RAND_MAX) - 1.0;
	}
	const auto test2 = test.detraceF();
	printf("%e %e %e \n", test2(0, 0, 3) , test2(0, 2, 1) , test2(2, 0, 1));

	return hpx::finalize();

	PRINT("Size of cuda_kick_shmem is %li\n", sizeof(cuda_kick_shmem));
	PRINT("cuda kick occupancy = %i\n", cuda_kick_occupancy());
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

int main(int argc, char *argv[]) {
	std::vector<std::string> cfg = {"hpx.commandline.allow_unknown=1"};
	cfg.push_back("hpx.stacks.small_size=2097152");
	hpx::init(argc, argv, cfg);
}

#endif
