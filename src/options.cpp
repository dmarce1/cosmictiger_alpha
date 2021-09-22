/*
 * options.cpp
 *
 *  Created on: Jan 25, 2021
 *      Author: dmarce1
 */

#include <cosmictiger/defs.hpp>
#include <cosmictiger/options.hpp>
#include <fstream>
#include <iostream>
#include <boost/program_options.hpp>
#include <cosmictiger/constants.hpp>
#include <cosmictiger/math.hpp>

bool process_options(int argc, char *argv[], options &opts) {
	namespace po = boost::program_options;
	bool rc = true;
	po::options_description command_opts("options");

	command_opts.add_options()                                                                       //
	("help", "produce help message")                                                                 //
	("omega_b", po::value<double>(&(opts.omega_b))->default_value(0.05), "") //
	("omega_c", po::value<double>(&(opts.omega_c))->default_value(0.26), "") //
	("Neff", po::value<double>(&(opts.Neff))->default_value(3.046), "") //
	("Theta", po::value<double>(&(opts.Theta))->default_value(1.00), "") //
	("Y", po::value<double>(&(opts.Y))->default_value(0.245), "") //
	("sigma8", po::value<double>(&(opts.sigma8))->default_value(0.8120), "") //
	("hubble", po::value<double>(&(opts.hubble))->default_value(0.673), "") //
	("ns", po::value<double>(&(opts.ns))->default_value(0.966), "spectral index") //
			;

	boost::program_options::variables_map vm;
	po::store(po::parse_command_line(argc, argv, command_opts), vm);
	po::notify(vm);
	if (vm.count("help")) {
		std::cout << command_opts << "\n";
		rc = false;
	}

	if (rc) {
		po::notify(vm);
		{
#define SHOW( opt ) PRINT( "%s = %e\n",  #opt, (double) opts.opt)
#define SHOW_STRING( opt ) std::cout << std::string( #opt ) << " = " << opts.opt << '\n';
		}
	}
	opts.omega_m = opts.omega_b + opts.omega_c;

	double omega_r = 32.0 * M_PI / 3.0 * constants::G * constants::sigma
			* (1 + opts.Neff * (7. / 8.0) * std::pow(4. / 11., 4. / 3.)) * std::pow(constants::H0, -2)
			* std::pow(constants::c, -3) * std::pow(2.73 * opts.Theta, 4) * std::pow(opts.hubble, -2);
	opts.omega_nu = omega_r * opts.Neff / (8.0 / 7.0 * std::pow(11.0 / 4.0, 4.0 / 3.0) + opts.Neff);
	opts.omega_gam = omega_r - opts.omega_nu;
	opts.omega_r = omega_r;

	PRINT("Simulation Options\n");

	SHOW(hubble);
	SHOW(ns);
	SHOW(sigma8);
	SHOW(Neff);
	SHOW(Y);
	SHOW(omega_m);
	SHOW(omega_r);
	SHOW(omega_c);
	SHOW(omega_b);
	SHOW(omega_gam);
	SHOW(omega_nu);
	SHOW(Theta);

	return rc;
}

