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
	std::string config;
	std::string test;
	std::string checkpt_file;
	size_t parts_dim;
	size_t nparts;
	int bucket_size;
	int map_size;
	int checkpt_freq;
	double hsoft;
	double theta;
	double code_to_s;
	double code_to_g;
	double code_to_cm;
	double code_to_cms;
	double omega_m;
	double hubble;
	double H0;
	double G;
	double z0;
	double M;
	double map_freq;
	double Y;
	double omega_b;
	double omega_c;
	double omega_gam;
	double omega_nu;
	double Neff;
	double Theta;
	double sigma8;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & config;
		arc & test;
		arc & checkpt_file;
		arc & parts_dim;
		arc & nparts;
		arc & bucket_size;
		arc & map_size;
		arc & checkpt_freq;
		arc & hsoft;
		arc & theta;
		arc & code_to_s;
		arc & code_to_g;
		arc & code_to_cm;
		arc & code_to_cms;
		arc & omega_m;
		arc & hubble;
		arc & H0;
		arc & G;
		arc & z0;
		arc & M;
		arc & map_freq;
		arc & Y;
		arc & omega_b;
		arc & omega_c;
		arc & omega_gam;
		arc & omega_nu;
		arc & Neff;
		arc & Theta;
		arc & sigma8;
	}
};

bool process_options(int argc, char *argv[], options &opts);                                          //

#endif /* COSMICTIGER_OPTIONS_HPP_ */
