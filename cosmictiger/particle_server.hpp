/*
 * particle_server.hpp
 *
 *  Created on: May 3, 2021
 *      Author: dmarce1
 */

#ifndef PARTICLE_SERVER_HPP_
#define PARTICLE_SERVER_HPP_



#include <cosmictiger/hpx.hpp>
#include <cosmictiger/particle_sets.hpp>

class particle_server {
	static particle_sets* parts;
	static size_t my_start;
	static size_t my_stop;
	static size_t my_size;
	static size_t global_size;
public:
	static void init();
};

#endif /* PARTICLE_SERVER_HPP_ */
