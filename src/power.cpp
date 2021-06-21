
#include <cosmictiger/power.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/constants.hpp>

HPX_PLAIN_ACTION(matter_power_spectrum_init);

void matter_power_spectrum(int filenum) {
	const int N = global().opts.parts_dim;
	const auto code_to_mpc = global().opts.code_to_cm / constants::mpc_to_cm;
	fourier3d_initialize(N);
	std::vector<hpx::future<void>> futs;
	for( int i = 0; i < hpx_size(); i++) {
		futs.push_back(hpx::async<matter_power_spectrum_init_action>(hpx_localities()[i]));
	}
	hpx::wait_all(futs.begin(),futs.end());
	fourier3d_inv_execute();
	auto spec = fourier3d_power_spectrum();
	const std::string filename = std::string("power.") + std::to_string(filenum) + std::string(".dat");
	FILE* fp = fopen( filename.c_str(), "wt");
	for( int n = 0; n < spec.size(); n++) {
		const float k = 2.0f * M_PI * n / code_to_mpc;
		fprintf( fp, "%e %e\n", k, spec[n] * std::pow(code_to_mpc, 3) );
	}
	fclose(fp);
	fourier3d_destroy();
}
