/*
 * options.hpp
 *
 *  Created on: Jan 25, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_OPTIONS_HPP_
#define COSMICTIGER_OPTIONS_HPP_

#include <string>

struct options {

	double hubble;
	double Neff;
	double ns;
	double omega_b;
	double omega_c;
	double omega_gam;
	double omega_m;
	double omega_nu;
	double omega_r;
	double sigma8;
	double Theta;
	double Y;
};

bool process_options(int argc, char *argv[], options &opts);                                          //

#endif /* COSMICTIGER_OPTIONS_HPP_ */
