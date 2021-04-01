/*
 * zeroverse.hpp
 *
 *  Created on: Mar 27, 2021
 *      Author: dmarce1
 */

#ifndef ZEROVERSE_HPP_
#define ZEROVERSE_HPP_

#include <cosmictiger/defs.hpp>

#include <functional>
#include <cosmictiger/vector.hpp>

template<class T>
struct interp_functor {
	vector<T> values;
	T amin;
	T amax;
	T minloga;
	T maxloga;
	int N;
	T dloga;

	T operator()(T a) const {
		T loga = logf(a);
		if (loga < minloga || loga > maxloga) {
			printf("Error in interpolation_function out of range %e %e %e\n", a, amin, amax);
		}
		int i1 = std::min(std::max(1, int((loga - minloga) / (dloga))), N - 3);
		int i0 = i1 - 1;
		int i2 = i1 + 1;
		int i3 = i2 + 1;
		const T c0 = values[i1];
		const T c1 = -values[i0] / (T) 3.0 - (T) 0.5 * values[i1] + values[i2] - values[i3] / (T) 6.0;
		const T c2 = (T) 0.5 * values[i0] - values[i1] + (T) 0.5 * values[i2];
		const T c3 = -values[i0] / (T) 6.0 + (T) 0.5 * values[i1] - (T) 0.5 * values[i2] + values[i3] / (T) 6.0;
		T x = (loga - i1 * dloga - minloga) / dloga;
		return c0 + c1 * x + c2 * x * x + c3 * x * x * x;
	}
};


struct zero_order_universe {
	double amin;
	double amax;
	std::function<float(float)> hubble;
	interp_functor<float> sigma_T;
	interp_functor<float> cs2;
	void compute_matter_fractions(float& Oc, float& Ob, float a) const;
	void compute_radiation_fractions(float& Ogam, float& Onu, float a) const;
	float conformal_time_to_scale_factor(float taumax);
	float scale_factor_to_conformal_time(float a);
	float redshift_to_time(float z) const;
	double redshift_to_density(double z) const;
};

void create_zero_order_universe(zero_order_universe* uni_ptr, double amax);



#endif /* ZEROVERSE_HPP_ */
