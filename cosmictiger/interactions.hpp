/*
 * interactions.hpp
 *
 *  Created on: Feb 6, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_INTERACTIONS_HPP_
#define COSMICTIGER_INTERACTIONS_HPP_

#include <cosmictiger/simd.hpp>
#include <cosmictiger/green_direct.hpp>
#include <cosmictiger/green_ewald.hpp>
// 986 // 251936
template<class T>
CUDA_EXPORT int multipole_interaction(expansion<T> &L, const multipole_type<T> &M, const expansion<T>& D, bool do_phi) { // 670/700 + 418 * NT + 50 * NFOUR
	sph_multipole_interaction(L,M,D);
	return 0;
}

// 516 / 251466
template<class T>
CUDA_EXPORT int multipole_interaction(array<T, NDIM + 1> &F, const multipole_type<T> &M, const expansion<T>& D, bool do_phi) { // 517 / 47428
	sphericalY<T,2> L;
	L = 0.0;
	sph_multipole_interaction(L,M,D);
	F[0] += L(0).real();
	F[1] += L(1,1).real();
	F[2] -= L(1,1).imag() * 0.707106781;
	F[3] -= L(1,0).real() * 0.707106781;
	return 0;
}

template<class T>
CUDA_EXPORT int multipole_interaction(expansion<T> &L, const T& m, const expansion<T>& D) { // 390 / 47301
	sphericalY<T,1> M;
	M(0).real() = m;
	M(0).imag() = T(0);
	sph_multipole_interaction(L,M,D);
	return 0;
}


template<class T>
CUDA_EXPORT int multipole_interaction(expansion<T> &L, const expansion<T>& D) { // 390 / 47301
	multipole_interaction(L,T(1.0),D);
	return 0;
}

#endif /* COSMICTIGER_INTERACTIONS_HPP_ */
