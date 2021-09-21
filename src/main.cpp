#include <cosmictiger/defs.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/initial.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/zero_order.hpp>

void yield() {
	hpx::this_thread::yield();
}

int hpx_main(int argc, char *argv[]) {
	options opts;
	if (process_options(argc, argv, opts)) {
		hpx_init();
		global_init(opts);
		initial_conditions();
	}
	return hpx::finalize();
}

#ifdef USE_HPX

int main(int argc, char *argv[]) {
	std::vector<std::string> cfg = { "hpx.commandline.allow_unknown=1" };
	cfg.push_back("hpx.stacks.small_size=2097152");
	hpx::init(argc, argv, cfg);
}

#endif
