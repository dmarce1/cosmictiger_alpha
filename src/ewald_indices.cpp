#include <cosmictiger/ewald_indices.hpp>
#include <cosmictiger/hpx.hpp>

HPX_PLAIN_ACTION(ewald_const::init, ewald_const_init_action);

void ewald_const::init() {
	hpx::future<void> futl, futr;
	auto children = hpx_child_localities();
	if (children.first != hpx::invalid_id) {
		futl = hpx::async<ewald_const_init_action>(children.first);
	}
	if (children.second != hpx::invalid_id) {
		futr = hpx::async<ewald_const_init_action>(children.second);
	}

	ewald_const::init_gpu();

	if (futl.valid()) {
		futl.get();
	}
	if (futr.valid()) {
		futr.get();
	}


}
