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
CUDA_EXPORT  int multipole_interaction(expansion<T> &L, const multipole_type<T> &M, const expansion<T>& D) { // 670/700 + 418 * NT + 50 * NFOUR
	int flops = 0;
	const auto half = (0.5f);
	const auto sixth = (1.0f / 6.0f);
	const auto halfD11 = half * D[11]; // 1
	const auto halfD12 = half * D[12]; // 1
	const auto halfD13 = half * D[13]; // 1
	const auto halfD15 = half * D[15]; // 1
	const auto halfD17 = half * D[17]; // 1
	const auto halfD18 = half * D[18]; // 1
	const auto halfD21 = half * D[21]; // 1
	const auto halfD22 = half * D[22]; // 1
	const auto halfD23 = half * D[23]; // 1
	const auto halfD24 = half * D[24]; // 1
	const auto halfD25 = half * D[25]; // 1
	const auto halfD26 = half * D[26]; // 1
	const auto halfD27 = half * D[27]; // 1
	const auto halfD28 = half * D[28]; // 1
	const auto halfD29 = half * D[29]; // 1
	const auto halfD31 = half * D[31]; // 1
	const auto halfD32 = half * D[32]; // 1
	const auto halfD33 = half * D[33]; // 1
	for (int i = 0; i < LP; i++) {
		L[i] = FMA(M[0], D[i], L[i]);   // 70
	}
	L[0] = FMA(M[1], D[4] * half, L[0]); // 3
	L[1] = FMA(M[1], D[10] * half, L[1]); // 3
	L[2] = FMA(M[1], halfD11, L[2]);     // 2
	L[3] = FMA(M[1], halfD12, L[3]);     // 2
	L[4] = FMA(M[1], D[20] * half, L[4]);     // 3
	L[5] = FMA(M[1], halfD21, L[5]);     // 2
	L[6] = FMA(M[1], halfD22, L[6]);     // 2
	L[7] = FMA(M[1], halfD23, L[7]);     // 2
	L[8] = FMA(M[1], halfD24, L[8]);     // 2
	L[9] = FMA(M[1], halfD25, L[9]);     // 2
	L[0] = FMA(M[2], D[5], L[0]);        // 2
	L[1] = FMA(M[2], D[11], L[1]);       // 2
	L[2] = FMA(M[2], D[13], L[2]);       // 2
	L[3] = FMA(M[2], D[14], L[3]);       // 2
	L[4] = FMA(M[2], D[21], L[4]);       // 2
	L[5] = FMA(M[2], D[23], L[5]);       // 2
	L[6] = FMA(M[2], D[24], L[6]);       // 2
	L[7] = FMA(M[2], D[26], L[7]);       // 2
	L[8] = FMA(M[2], D[27], L[8]);       // 2
	L[9] = FMA(M[2], D[28], L[9]);       // 2
	L[0] = FMA(M[3], D[6], L[0]);        // 2
	L[1] = FMA(M[3], D[12], L[1]);       // 2
	L[2] = FMA(M[3], D[14], L[2]);       // 2
	L[3] = FMA(M[3], D[15], L[3]);       // 2
	L[4] = FMA(M[3], D[22], L[4]);       // 2
	L[5] = FMA(M[3], D[24], L[5]);       // 2
	L[6] = FMA(M[3], D[25], L[6]);       // 2
	L[7] = FMA(M[3], D[27], L[7]);       // 2
	L[8] = FMA(M[3], D[28], L[8]);       // 2
	L[9] = FMA(M[3], D[29], L[9]);       // 2
	L[0] = FMA(M[4], D[7] * half, L[0]); // 3
	L[1] = FMA(M[4], halfD13, L[1]);     // 2
	L[2] = FMA(M[4], D[16] * half, L[2]);     // 3
	L[3] = FMA(M[4], halfD17, L[3]);     // 2
	L[4] = FMA(M[4], halfD23, L[4]);     // 2
	L[5] = FMA(M[4], halfD26, L[5]);     // 2
	L[6] = FMA(M[4], halfD27, L[6]);     // 2
	L[7] = FMA(M[4], D[30] * half, L[7]);     // 3
	L[8] = FMA(M[4], halfD31, L[8]);     // 2
	L[9] = FMA(M[4], halfD32, L[9]);     // 2
	L[0] = FMA(M[5], D[8], L[0]);        // 2
	L[1] = FMA(M[5], D[14], L[1]);       // 2
	L[2] = FMA(M[5], D[17], L[2]);       // 2
	L[3] = FMA(M[5], D[18], L[3]);       // 2
	L[4] = FMA(M[5], D[24], L[4]);       // 2
	L[5] = FMA(M[5], D[27], L[5]);       // 2
	L[6] = FMA(M[5], D[28], L[6]);       // 2
	L[7] = FMA(M[5], D[31], L[7]);       // 2
	L[8] = FMA(M[5], D[32], L[8]);       // 2
	L[9] = FMA(M[5], D[33], L[9]);       // 2
	L[0] = FMA(M[6], D[9] * half, L[0]); // 3
	L[1] = FMA(M[6], halfD15, L[1]);     // 2
	L[2] = FMA(M[6], halfD18, L[2]);     // 2
	L[3] = FMA(M[6], D[19] * half, L[3]);     // 3
	L[4] = FMA(M[6], halfD25, L[4]);     // 2
	L[5] = FMA(M[6], halfD28, L[5]);     // 2
	L[6] = FMA(M[6], halfD29, L[6]);     // 2
	L[7] = FMA(M[6], halfD32, L[7]);     // 2
	L[8] = FMA(M[6], halfD33, L[8]);     // 2
	L[9] = FMA(M[6], D[34] * half, L[9]);     // 3
	L[0] = FMA(M[7], D[10] * sixth, L[0]);     //3
	L[1] = FMA(M[7], D[20] * sixth, L[1]);     //3
	L[2] = FMA(M[7], D[21] * sixth, L[2]);     //3
	L[3] = FMA(M[7], D[22] * sixth, L[3]);     //3
	L[0] = FMA(M[8], halfD11, L[0]);     // 2
	L[1] = FMA(M[8], halfD21, L[1]);     // 2
	L[2] = FMA(M[8], halfD23, L[2]);     // 2
	L[3] = FMA(M[8], halfD24, L[3]);     // 2
	L[0] = FMA(M[9], halfD12, L[0]);     // 2
	L[1] = FMA(M[9], halfD22, L[1]);     // 2
	L[2] = FMA(M[9], halfD24, L[2]);     // 2
	L[3] = FMA(M[9], halfD25, L[3]);     // 2
	L[0] = FMA(M[10], halfD13, L[0]);    // 2
	L[1] = FMA(M[10], halfD23, L[1]);    // 2
	L[2] = FMA(M[10], halfD26, L[2]);    // 2
	L[3] = FMA(M[10], halfD27, L[3]);    // 2
	L[0] = FMA(M[11], D[14], L[0]);      // 2
	L[1] = FMA(M[11], D[24], L[1]);      // 2
	L[2] = FMA(M[11], D[27], L[2]);      // 2
	L[3] = FMA(M[11], D[28], L[3]);      // 2
	L[0] = FMA(M[12], halfD15, L[0]);    // 2
	L[1] = FMA(M[12], halfD25, L[1]);    // 2
	L[2] = FMA(M[12], halfD28, L[2]);    // 2
	L[3] = FMA(M[12], halfD29, L[3]);    // 2
	L[0] = FMA(M[13], D[16] * sixth, L[0]);    //3
	L[1] = FMA(M[13], D[26] * sixth, L[1]);    //3
	L[2] = FMA(M[13], D[30] * sixth, L[2]);    //3
	L[3] = FMA(M[13], D[31] * sixth, L[3]);    //3
	L[0] = FMA(M[14], halfD17, L[0]);     // 2
	L[1] = FMA(M[14], halfD27, L[1]);     // 2
	L[2] = FMA(M[14], halfD31, L[2]);     // 2
	L[3] = FMA(M[14], halfD32, L[3]);     // 2
	L[0] = FMA(M[15], halfD18, L[0]);     // 2
	L[1] = FMA(M[15], halfD28, L[1]);     // 2
	L[2] = FMA(M[15], halfD32, L[2]);     // 2
	L[3] = FMA(M[15], halfD33, L[3]);     // 2
	L[0] = FMA(M[16], D[19] * sixth, L[0]);     // 3
	L[1] = FMA(M[16], D[29] * sixth, L[1]);     // 3
	L[2] = FMA(M[16], D[33] * sixth, L[2]);     // 3
	L[3] = FMA(M[16], D[34] * sixth, L[3]);     // 3
	return flops + 309;
}

// 516 / 251466
template<class T>
CUDA_EXPORT  int multipole_interaction(array<T, NDIM + 1> &L, const multipole_type<T> &M, const expansion<T>& D) { // 517 / 47428

	int flops = 0;
	flops += 1 + NDIM;
	const auto half = T(0.5);
	const auto sixth = T(1.0 / 6.0);
	const auto halfD11 = half * D[11];            // 1
	const auto halfD12 = half * D[12];            // 1
	const auto halfD13 = half * D[13];            // 1
	const auto halfD15 = half * D[15];            // 1
	const auto halfD17 = half * D[17];            // 1
	const auto halfD18 = half * D[18];            // 1
	const auto halfD21 = half * D[21];            // 1
	const auto halfD22 = half * D[22];            // 1
	const auto halfD23 = half * D[23];            // 1
	const auto halfD24 = half * D[24];            // 1
	const auto halfD25 = half * D[25];            // 1
	const auto halfD26 = half * D[26];            // 1
	const auto halfD27 = half * D[27];            // 1
	const auto halfD28 = half * D[28];            // 1
	const auto halfD29 = half * D[29];            // 1
	const auto halfD31 = half * D[31];            // 1
	const auto halfD32 = half * D[32];            // 1
	const auto halfD33 = half * D[33];            // 1
	for (int i = 0; i < NDIM + 1; i++) {
		L[i] = FMA(M[0], D[i], L[i]);              // 8
	}
	L[0] = FMA(M[1], D[4] * half, L[0]);          // 3
	L[1] = FMA(M[1], D[10] * half, L[1]);         // 3
	L[2] = FMA(M[1], halfD11, L[2]);              // 2
	L[3] = FMA(M[1], halfD12, L[3]);              // 2
	L[0] = FMA(M[2], D[5], L[0]);                 // 2
	L[1] = FMA(M[2], D[11], L[1]);                // 2
	L[2] = FMA(M[2], D[13], L[2]);                // 2
	L[3] = FMA(M[2], D[14], L[3]);                // 2
	L[0] = FMA(M[3], D[6], L[0]);                 // 2
	L[1] = FMA(M[3], D[12], L[1]);                // 2
	L[2] = FMA(M[3], D[14], L[2]);                // 2
	L[3] = FMA(M[3], D[15], L[3]);                // 2
	L[0] = FMA(M[4], D[7] * half, L[0]);          // 3
	L[1] = FMA(M[4], halfD13, L[1]);              // 2
	L[2] = FMA(M[4], D[16] * half, L[2]);         // 3
	L[3] = FMA(M[4], halfD17, L[3]);              // 2
	L[0] = FMA(M[5], D[8], L[0]);                 // 2
	L[1] = FMA(M[5], D[14], L[1]);                // 2
	L[2] = FMA(M[5], D[17], L[2]);                // 2
	L[3] = FMA(M[5], D[18], L[3]);                // 2
	L[0] = FMA(M[6], D[9] * half, L[0]);          // 3
	L[1] = FMA(M[6], halfD15, L[1]);              // 2
	L[2] = FMA(M[6], halfD18, L[2]);              // 2
	L[3] = FMA(M[6], D[19] * half, L[3]);         // 3
	L[0] = FMA(M[7], D[10] * sixth, L[0]);        // 3
	L[1] = FMA(M[7], D[20] * sixth, L[1]);        // 3
	L[2] = FMA(M[7], D[21] * sixth, L[2]);        // 3
	L[3] = FMA(M[7], D[22] * sixth, L[3]);        // 3
	L[0] = FMA(M[8], halfD11, L[0]);              // 2
	L[1] = FMA(M[8], halfD21, L[1]);              // 2
	L[2] = FMA(M[8], halfD23, L[2]);              // 2
	L[3] = FMA(M[8], halfD24, L[3]);              // 2
	L[0] = FMA(M[9], halfD12, L[0]);              // 2
	L[1] = FMA(M[9], halfD22, L[1]);              // 2
	L[2] = FMA(M[9], halfD24, L[2]);              // 2
	L[3] = FMA(M[9], halfD25, L[3]);              // 2
	L[0] = FMA(M[10], halfD13, L[0]);             // 2
	L[1] = FMA(M[10], halfD23, L[1]);             // 2
	L[2] = FMA(M[10], halfD26, L[2]);             // 2
	L[3] = FMA(M[10], halfD27, L[3]);             // 2
	L[0] = FMA(M[11], D[14], L[0]);               // 2
	L[1] = FMA(M[11], D[24], L[1]);               // 2
	L[2] = FMA(M[11], D[27], L[2]);               // 2
	L[3] = FMA(M[11], D[28], L[3]);               // 2
	L[0] = FMA(M[12], halfD15, L[0]);             // 2
	L[1] = FMA(M[12], halfD25, L[1]);             // 2
	L[2] = FMA(M[12], halfD28, L[2]);             // 2
	L[3] = FMA(M[12], halfD29, L[3]);             // 2
	L[0] = FMA(M[13], D[16] * sixth, L[0]);       // 3
	L[1] = FMA(M[13], D[26] * sixth, L[1]);       // 3
	L[2] = FMA(M[13], D[30] * sixth, L[2]);       // 3
	L[3] = FMA(M[13], D[31] * sixth, L[3]);       // 3
	L[0] = FMA(M[14], halfD17, L[0]);             // 2
	L[1] = FMA(M[14], halfD27, L[1]);             // 2
	L[2] = FMA(M[14], halfD31, L[2]);             // 2
	L[3] = FMA(M[14], halfD32, L[3]);             // 2
	L[0] = FMA(M[15], halfD18, L[0]);             // 2
	L[1] = FMA(M[15], halfD28, L[1]);             // 2
	L[2] = FMA(M[15], halfD32, L[2]);             // 2
	L[3] = FMA(M[15], halfD33, L[3]);             // 2
	L[0] = FMA(M[16], D[19] * sixth, L[0]);       // 3
	L[1] = FMA(M[16], D[29] * sixth, L[1]);       // 3
	L[2] = FMA(M[16], D[33] * sixth, L[2]);       // 3
	L[3] = FMA(M[16], D[34] * sixth, L[3]);       // 3
	return 172;
}

template<class T>
CUDA_EXPORT  int multipole_interaction(expansion<T> &L, const expansion<T>& D) { // 390 / 47301
	for (int i = 0; i < LP; i++) {
		L[i] += D[i];
	}
	return 35;
}

template<class T>
CUDA_EXPORT  int multipole_interaction(expansion<T> &L, const T& M, const expansion<T>& D) { // 390 / 47301
	for (int i = 0; i < LP; i++) {
		L[i] = FMA(M, D[i], L[i]);
		;
	}
	return 70;
}

#endif /* COSMICTIGER_INTERACTIONS_HPP_ */
