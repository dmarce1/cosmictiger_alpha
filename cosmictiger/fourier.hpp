/*
 * fourier.hpp
 *
 *  Created on: Jan 12, 2021
 *      Author: dmarce1
 */

#ifndef FOURIER_HPP_
#define FOURIER_HPP_

#include <cosmictiger/math.hpp>



void fft1d(std::vector<cmplx>& Y, int N);
void fft2d(std::vector<cmplx>& Y, int N);

void fourier3d_initialize(int N_);
void fourier3d_destroy();
void fourier3d_execute();
void fourier3d_inv_execute();
void fourier3d_accumulate_real(int xb, int xe, int yb, int ye, int zb, int ze, std::vector<float> data);
void fourier3d_accumulate(int xb, int xe, int yb, int ye, int zb, int ze, std::vector<cmplx> data);
std::vector<cmplx> fourier3d_read(int xb, int xe, int yb, int ye, int zb, int ze);
std::vector<float> fourier3d_read_real(int xb, int xe, int yb, int ye, int zb, int ze);
void fourier3d_mirror();
std::vector<float> fourier3d_power_spectrum();


#endif /* FOURIER_HPP_ */
