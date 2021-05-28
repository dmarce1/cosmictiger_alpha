/*
 * fourier.hpp
 *
 *  Created on: Jan 12, 2021
 *      Author: dmarce1
 */

#ifndef FOURIER_HPP_
#define FOURIER_HPP_

#include <cosmictiger/math.hpp>



void fft3d(cmplx* Y, int N);
void fft2d(cmplx* Y, int N);


void fft3d_inv(cmplx* Y, int N);
void fft32_inv(cmplx* Y, int N);


#endif /* FOURIER_HPP_ */
