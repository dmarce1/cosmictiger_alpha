/*
 * chemsitry.hpp
 *
 *  Created on: Jan 10, 2021
 *      Author: dmarce1
 */

#ifndef GPUTIGER_CHEMISTRY_HPP_
#define GPUTIGER_CHEMISTRY_HPP_

#include <cosmictiger/constants.hpp>
#include <cosmictiger/math.hpp>
#include <functional>


void chemistry_update(const std::function<double(double)> &Hubble, double &H, double &Hp, double &He, double &Hep, double &Hepp,
		double &ne, double T, double a, double dt);
#endif /* GPUTIGER_CHEMISTRY_HPP_ */
