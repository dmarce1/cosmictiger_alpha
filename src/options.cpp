/*
 * options.cpp
 *
 *  Created on: Jan 25, 2021
 *      Author: dmarce1
 */

#include <cosmictiger/defs.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/hpx.hpp>
#include <fstream>
#include <boost/program_options.hpp>
#include <cosmictiger/constants.hpp>

bool process_options(int argc, char *argv[], options &opts) {
	namespace po = boost::program_options;
	bool rc;
	po::options_description command_opts("options");

	command_opts.add_options()                                                                       //
	("help", "produce help message")                                                                 //
	("config", po::value<std::string>(&(opts.config))->default_value(""), "configuration file") //
	("bucket_size", po::value<int>(&(opts.bucket_size))->default_value(MAX_BUCKET_SIZE), "bucket size") //
	("checkpt_freq", po::value<int>(&(opts.checkpt_freq))->default_value(3600), "checkpoint frequency") //
	("checkpt_file", po::value<std::string>(&(opts.checkpt_file))->default_value(""), "checkpoint restart") //
	("cuda", po::value<bool>(&(opts.cuda))->default_value(true), "cuda on/off") //
	("code_to_g", po::value<double>(&(opts.code_to_g))->default_value(1.99e33), "code to g") //
	("code_to_cm", po::value<double>(&(opts.code_to_cm))->default_value(6.17e27), "code to cm") //
	("code_to_cms", po::value<double>(&(opts.code_to_cms))->default_value(3e10), "code to cm/s") //
	("omega_m", po::value<double>(&(opts.omega_m))->default_value(0.32), "") //
	("hubble", po::value<double>(&(opts.hubble))->default_value(0.7), "") //
	("silo_interval", po::value<double>(&(opts.silo_interval))->default_value(-1.), "interval between SILO outs") //
	("map_size", po::value<int>(&(opts.map_size))->default_value(-1), "smallest dimension of Mollweide map (-1 = off)") //
	("map_freq", po::value<double>(&(opts.map_freq))->default_value(1. / 128.),
			"Map frequency, conformal time, 1 = age of universe") //
	("parts_dim", po::value<size_t>(&(opts.parts_dim))->default_value(16), "number of particles = parts_dim^3") //
	("test", po::value<std::string>(&(opts.test))->default_value(""), "test problem") //
			;

	boost::program_options::variables_map vm;
	po::store(po::parse_command_line(argc, argv, command_opts), vm);
	po::notify(vm);
	if (vm.count("help")) {
		std::cout << command_opts << "\n";
		rc = false;
	} else {
		if (!opts.config.empty()) {
			std::ifstream cfg_fs { vm["config"].as<std::string>() };
			if (cfg_fs) {
				po::store(po::parse_config_file(cfg_fs, command_opts), vm);
				rc = true;
			} else {
				printf("Configuration file %s not found!\n", opts.config.c_str());
				rc = false;
			}
		} else {
			rc = true;
		}
	}
	if (rc) {
		po::notify(vm);
		opts.nparts = opts.parts_dim * opts.parts_dim * opts.parts_dim;
		{
#define SHOW( opt ) std::cout << std::string( #opt ) << " = " << std::to_string(opts.opt) << '\n';
#define SHOW_STRING( opt ) std::cout << std::string( #opt ) << " = " << opts.opt << '\n';
			SHOW_STRING(config);
			SHOW_STRING(test);
			SHOW(nparts);
		}
	}
	opts.hsoft = 1.0 / pow(opts.nparts, 1.0 / 3.0) / 25.0;
	opts.theta = 0.7;
	opts.G = opts.M = 1.0;
	if (opts.bucket_size > MAX_BUCKET_SIZE) {
		printf("Bucket size of %i exceeds max of %i\n", opts.bucket_size, MAX_BUCKET_SIZE);
		abort();
	}
	opts.Neff = 3.046;
	opts.Y = 0.24;
	opts.omega_b = 0.05;
	opts.omega_c = 0.25;
	opts.Theta = 1.0;
	opts.sigma8 = 0.83;
	double omega_r = 32.0 * M_PI / 3.0 * constants::G * constants::sigma
			* (1 + opts.Neff * (7. / 8.0) * std::pow(4. / 11., 4. / 3.)) * std::pow(constants::H0, -2)
			* std::pow(constants::c, -3) * std::pow(2.73 * opts.Theta, 4) * std::pow(opts.hubble, -2);
	opts.omega_nu = omega_r * opts.Neff / (8.0 / 7.0 * std::pow(11.0 / 4.0, 4.0 / 3.0) + opts.Neff);
	opts.omega_gam = omega_r - opts.omega_nu;

	return rc;
}

