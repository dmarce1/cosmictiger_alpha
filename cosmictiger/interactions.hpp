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
	//L = L + interaction<T, LORDER, MORDER>(M, D);
	direct_interaction<T>(L,M,D);
	return 0;
}

// 516 / 251466
template<class T>
CUDA_EXPORT int multipole_interaction(array<T, NDIM + 1> &L, const multipole_type<T> &M, const expansion<T>& D,
		bool do_phi) { // 517 / 47428
	const auto L0 = interaction<T, 2, MORDER>(M, D);
	L[0] += L0(0, 0, 0);
	L[1] += L0(1, 0, 0);
	L[2] += L0(0, 1, 0);
	L[3] += L0(0, 0, 1);
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
