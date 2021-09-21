#include <cosmictiger/defs.hpp>
#include <cosmictiger/initial.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/global.hpp>
#include <cosmictiger/zero_order.hpp>


int main(int argc, char *argv[]) {
	options opts;
	if (process_options(argc, argv, opts)) {
		global_init(opts);
		initial_conditions();
	}
	return 0;
}
