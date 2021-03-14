/*
 * options.cpp
 *
 *  Created on: Jan 25, 2021
 *      Author: dmarce1
 */

#include <cosmictiger/options.hpp>
#include <cosmictiger/hpx.hpp>
#include <fstream>
#include <boost/program_options.hpp>

bool process_options(int argc, char *argv[], options &opts) {
	namespace po = boost::program_options;
	bool rc;
	po::options_description command_opts("options");

	command_opts.add_options()                                                                       //
	("help", "produce help message")                                                                 //
	("config", po::value<std::string>(&(opts.config))->default_value(""), "configuration file") //
	("code_to_g", po::value<double>(&(opts.code_to_g))->default_value(1.99e43), "code to g") //
	("code_to_cm", po::value<double>(&(opts.code_to_cm))->default_value(6.17e27), "code to cm") //
	("code_to_cms", po::value<double>(&(opts.code_to_cms))->default_value(3e10), "code to cm/s") //
	("omega_m", po::value<double>(&(opts.omega_m))->default_value(0.32), "") //
	("hubble", po::value<double>(&(opts.hubble))->default_value(0.7), "") //
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
	opts.hsoft = 1.0 / pow(opts.nparts, 1.0 / 3.0) / 50.0;
	opts.theta = 0.7;

	const auto Gcgs = 6.67259e-8;
	const auto ccgs = 2.99792458e+10;
	const auto Hcgs = 3.2407789e-18;
	opts.code_to_s = opts.code_to_cm / opts.code_to_cms;
	opts.H0 = Hcgs * opts.code_to_s;
	opts.G = Gcgs / pow(opts.code_to_cm, 3) * opts.code_to_g * pow(opts.code_to_s, 2);
	double m_tot = opts.omega_m * 3.0 * opts.H0 * opts.H0 / (8 * M_PI * opts.G);
	opts.M = m_tot / opts.nparts;
	printf("G in code units = %e\n", opts.G);
	printf("M in code units = %e\n", opts.M);
	return rc;
}

