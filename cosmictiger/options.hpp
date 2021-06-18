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

	int bucket_size;
	int checkpt_freq;
	int map_size;

	bool cuda;
	bool glass;
	bool groups;
	bool power;
	bool maps;

	double code_to_cm;
	double code_to_cms;
	double code_to_g;
	double code_to_s;
	double G;
	double H0;
	double hsoft;
	double hubble;
	double M;
	double map_freq;
	double Neff;
	double omega_b;
	double omega_c;
	double omega_gam;
	double omega_m;
	double omega_nu;
	double omega_r;
	double sigma8;
	double silo_interval;
	double theta;
	double Theta;
	double z0;
	double z1;

	size_t nparts;
	size_t parts_dim;

	std::string checkpt_file;
	std::string config;
	std::string glass_file;
	std::string test;

	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & bucket_size;
		arc & checkpt_freq;
		arc & map_size;
		arc & cuda;
		arc & glass;
		arc & groups;
		arc & power;
		arc & maps;
		arc & code_to_cm;
		arc & code_to_cms;
		arc & code_to_g;
		arc & code_to_s;
		arc & G;
		arc & H0;
		arc & hsoft;
		arc & hubble;
		arc & M;
		arc & map_freq;
		arc & Neff;
		arc & omega_b;
		arc & omega_c;
		arc & omega_gam;
		arc & omega_m;
		arc & omega_nu;
		arc & omega_r;
		arc & sigma8;
		arc & silo_interval;
		arc & theta;
		arc & Theta;
		arc & z0;
		arc & z1;
		arc & nparts;
		arc & parts_dim;
		arc & checkpt_file;
		arc & config;
		arc & glass_file;
		arc & test;
	}
};

bool process_options(int argc, char *argv[], options &opts);                                          //

#endif /* COSMICTIGER_OPTIONS_HPP_ */
