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
#include <iostream>
#ifdef USE_HPX
#else
#include <boost/program_options.hpp>
#endif
#include <cosmictiger/constants.hpp>
#include <cosmictiger/math.hpp>

bool process_options(int argc, char *argv[], options &opts) {
#ifdef USE_HPX
	namespace po = hpx::program_options;
#else
	namespace po = boost::program_options;
#endif
	bool rc;
	po::options_description command_opts("options");

	command_opts.add_options()                                                                       //
	("help", "produce help message")                                                                 //
	("bucket_size", po::value<int>(&(opts.bucket_size))->default_value(MAX_BUCKET_SIZE), "bucket size") //
	("checkpt_freq", po::value<int>(&(opts.checkpt_freq))->default_value(3600), "checkpoint frequency") //
	("checkpt_file", po::value<std::string>(&(opts.checkpt_file))->default_value(""), "checkpoint restart") //
	("config", po::value<std::string>(&(opts.config))->default_value(""), "configuration file") //
	("cuda", po::value<bool>(&(opts.cuda))->default_value(true), "cuda on/off") //
	("glass_file", po::value<std::string>(&(opts.glass_file))->default_value(""), "glass checkpoint for IC") //
	("glass", po::value<bool>(&(opts.glass))->default_value(false), "produce glass file") //
	("groups", po::value<bool>(&(opts.groups))->default_value(false), "do groups") //
	("power", po::value<bool>(&(opts.power))->default_value(false), "do mass power spectrum") //
	("code_to_g", po::value<double>(&(opts.code_to_g))->default_value(1.99e33), "code to g") //
	("code_to_cm", po::value<double>(&(opts.code_to_cm))->default_value(6.17e27), "code to cm") //
	("code_to_cms", po::value<double>(&(opts.code_to_cms))->default_value(3e10), "code to cm/s") //
	("omega_b", po::value<double>(&(opts.omega_b))->default_value(0.049), "") //
	("omega_c", po::value<double>(&(opts.omega_c))->default_value(0.265), "") //
	("Neff", po::value<double>(&(opts.Neff))->default_value(3.046), "") //
	("Theta", po::value<double>(&(opts.Theta))->default_value(1.00), "") //
	("sigma8", po::value<double>(&(opts.sigma8))->default_value(0.81), "") //
	("theta", po::value<double>(&(opts.theta))->default_value(0.7), "") //
	("hubble", po::value<double>(&(opts.hubble))->default_value(.673), "") //
	("silo_interval", po::value<double>(&(opts.silo_interval))->default_value(-1.), "interval between SILO outs") //
	("maps", po::value<bool>(&(opts.maps))->default_value(false), "generate healpix maps") //
	("parts_dim", po::value<size_t>(&(opts.parts_dim))->default_value(128), "number of particles = parts_dim^3") //
	("test", po::value<std::string>(&(opts.test))->default_value(""), "test problem") //
	("z0", po::value<double>(&(opts.z0))->default_value(49), "starting redshift") //
			;

#ifdef USE_HPX
	hpx::program_options::variables_map vm;
#else
	boost::program_options::variables_map vm;
#endif
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
				PRINT("Configuration file %s not found!\n", opts.config.c_str());
				return false;
			}
		} else {
			rc = true;
		}
	}

	if (rc) {
		po::notify(vm);
		opts.nparts = opts.parts_dim * opts.parts_dim * opts.parts_dim;
		{
#define SHOW( opt ) PRINT( "%s = %e\n",  #opt, (double) opts.opt)
#define SHOW_STRING( opt ) std::cout << std::string( #opt ) << " = " << opts.opt << '\n';
		}
	}
	opts.hsoft = 1.0 / pow(opts.nparts, 1.0 / 3.0) / 25.0;

	if (opts.bucket_size > MAX_BUCKET_SIZE) {
		PRINT("Bucket size of %i exceeds max of %i\n", opts.bucket_size, MAX_BUCKET_SIZE);
		abort();
	}
	opts.omega_m = opts.omega_b + opts.omega_c;
	// pkdgrav3 run dims
	opts.code_to_cm = 7.108e26 * opts.parts_dim / 1024.0 / opts.hubble;
	// millenial run dims
	//opts.code_to_cm = 1.8921460945e+27 * opts.parts_dim / 2160.0 / opts.hubble;
	//opts.code_to_cm = constants::mpc_to_cm * 1000.0;

	const auto Gcgs = constants::G;
	const auto ccgs = constants::c;
	const auto Hcgs = constants::H0;
	opts.code_to_s = opts.code_to_cm / opts.code_to_cms;
	opts.H0 = Hcgs * opts.code_to_s;
	opts.G = Gcgs / pow(opts.code_to_cm, 3) * opts.code_to_g * pow(opts.code_to_s, 2);
	double m_tot = (opts.glass ? 1.0 : opts.omega_m) * 3.0 * sqr(opts.H0 * opts.hubble) / (8 * M_PI * opts.G);
	opts.M = m_tot / opts.nparts;

	if (opts.glass) {
		opts.G = -opts.G;
	}

	double omega_r = 32.0 * M_PI / 3.0 * constants::G * constants::sigma
			* (1 + opts.Neff * (7. / 8.0) * std::pow(4. / 11., 4. / 3.)) * std::pow(constants::H0, -2)
			* std::pow(constants::c, -3) * std::pow(2.73 * opts.Theta, 4) * std::pow(opts.hubble, -2);
	opts.omega_nu = omega_r * opts.Neff / (8.0 / 7.0 * std::pow(11.0 / 4.0, 4.0 / 3.0) + opts.Neff);
	opts.omega_gam = omega_r - opts.omega_nu;
	opts.omega_r = omega_r;

	if (opts.maps) {
		opts.map_size = 2 * opts.parts_dim;
	} else {
		opts.map_size = -1;
	}
	opts.map_freq = 0.01;
	PRINT("Simulation Options\n");

	SHOW_STRING(config);
	SHOW(map_size);
	SHOW(silo_interval);
	SHOW(maps);
	SHOW(checkpt_freq);
	SHOW_STRING(checkpt_file);
	SHOW_STRING(test);
	SHOW(parts_dim);
	SHOW(nparts);
	SHOW(bucket_size);
	SHOW(cuda);
	SHOW(groups);
	SHOW(hsoft);
	SHOW(power);
	SHOW(theta);

	PRINT("Units\n");
	SHOW(code_to_s);
	SHOW(code_to_g);
	SHOW(code_to_cm);
	SHOW(code_to_cms);
	SHOW(H0);
	SHOW(G);
	SHOW(M);

	PRINT("Cosmological Options\n");
	SHOW(hubble);
	SHOW(z0);
	SHOW(sigma8);
	SHOW(Neff);
	SHOW(omega_m);
	SHOW(omega_r);
	SHOW(omega_c);
	SHOW(omega_b);
	SHOW(omega_gam);
	SHOW(omega_nu);
	SHOW(Theta);

	return rc;
}

