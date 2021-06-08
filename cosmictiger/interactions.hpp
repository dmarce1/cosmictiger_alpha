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

	return 0;
}

// 516 / 251466
template<class T>
CUDA_EXPORT int multipole_interaction(array<T, NDIM + 1> &L, const multipole_type<T> &M, const expansion<T>& D, bool do_phi) { // 517 / 47428
	return 0;

}

template<class T>
CUDA_EXPORT int multipole_interaction(expansion<T> &L, const expansion<T>& D) { // 390 / 47301
	for (int i = 0; i < LP; i++) {
		L[i] += D[i];
	}
	return 35;
}

template<class T>
CUDA_EXPORT int multipole_interaction(expansion<T> &L, const T& M, const expansion<T>& D) { // 390 / 47301
	for (int i = 0; i < LP; i++) {
		L[i] = FMA(M, D[i], L[i]);
		;
	}
	return 70;
}

#endif /* COSMICTIGER_INTERACTIONS_HPP_ */
