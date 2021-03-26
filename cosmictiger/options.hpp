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
	size_t parts_dim;
	size_t nparts;
	int bucket_size;
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
	int map_size;
	double map_freq;
	int checkpt_freq;
	std::string checkpt_file;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & config;
		arc & test;
		arc & nparts;
	}
};

bool process_options(int argc, char *argv[], options &opts);                                          //

#endif /* COSMICTIGER_OPTIONS_HPP_ */
