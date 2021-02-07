/*
 * interactions.hpp
 *
 *  Created on: Feb 6, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_INTERACTIONS_HPP_
#define COSMICTIGER_INTERACTIONS_HPP_

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/expansion.hpp>

#ifndef __CUDA_ARCH__
template<class T>
CUDA_EXPORT inline T fmaf(T a, T b, T c) {
   return a * b + c;
}
#endif

template<class T>  // 167
CUDA_EXPORT void green_deriv_direct(expansion &D, const T &d0, const T &d1, const T &d2, const T &d3, const T &d4,
      const array<T, NDIM> &dx) {
   T threedxadxb;
   T dxadxbdxc;
   const auto dx0dx0 = dx[0] * dx[0];
   const auto dx0dx1 = dx[0] * dx[1];
   const auto dx0dx2 = dx[0] * dx[2];
   const auto dx1dx1 = dx[1] * dx[1];
   const auto dx1dx2 = dx[1] * dx[2];
   const auto dx2dx2 = dx[2] * dx[2];
   const auto &dx1dx0 = dx0dx1;
   const auto &dx2dx0 = dx0dx2;
   const auto &dx2dx1 = dx1dx2;
   D[0] = d0;
   D[1] = dx[0] * d1;
   D[4] = dx0dx0 * d2;
   dxadxbdxc = dx0dx0 * dx[0];
   D[10] = dxadxbdxc * d3;
   D[20] = dxadxbdxc * dx[0] * d4;
   D[2] = dx[1] * d1;
   D[5] = dx1dx0 * d2;
   dxadxbdxc = dx1dx0 * dx[0];
   D[11] = dxadxbdxc * d3;
   D[21] = dxadxbdxc * dx[0] * d4;
   D[7] = dx1dx1 * d2;
   dxadxbdxc = dx1dx1 * dx[0];
   D[13] = dxadxbdxc * d3;
   D[23] = dxadxbdxc * dx[0] * d4;
   dxadxbdxc = dx1dx1 * dx[1];
   D[16] = dxadxbdxc * d3;
   D[26] = dxadxbdxc * dx[0] * d4;
   D[30] = dxadxbdxc * dx[1] * d4;
   D[3] = dx[2] * d1;
   D[6] = dx2dx0 * d2;
   dxadxbdxc = dx2dx0 * dx[0];
   D[12] = dxadxbdxc * d3;
   D[22] = dxadxbdxc * dx[0] * d4;
   D[8] = dx2dx1 * d2;
   dxadxbdxc = dx2dx1 * dx[0];
   D[14] = dxadxbdxc * d3;
   D[24] = dxadxbdxc * dx[0] * d4;
   dxadxbdxc = dx2dx1 * dx[1];
   D[17] = dxadxbdxc * d3;
   D[27] = dxadxbdxc * dx[0] * d4;
   D[31] = dxadxbdxc * dx[1] * d4;
   D[9] = dx2dx2 * d2;
   dxadxbdxc = dx2dx2 * dx[0];
   D[15] = dxadxbdxc * d3;
   D[25] = dxadxbdxc * dx[0] * d4;
   dxadxbdxc = dx2dx2 * dx[1];
   D[18] = dxadxbdxc * d3;
   D[28] = dxadxbdxc * dx[0] * d4;
   D[32] = dxadxbdxc * dx[1] * d4;
   dxadxbdxc = dx2dx2 * dx[2];
   D[19] = dxadxbdxc * d3;
   D[29] = dxadxbdxc * dx[0] * d4;
   D[33] = dxadxbdxc * dx[1] * d4;
   D[34] = dxadxbdxc * dx[2] * d4;

   const auto dx0d2 = dx[0] * d2;
   const auto dx1d2 = dx[1] * d2;
   const auto dx2d2 = dx[2] * d2;
   D[4] += d1;
   D[10] = fmaf(float(3), dx0d2, D[10]);
   D[20] = fmaf(float(6) * dx0dx0, d3, D[20]);
   D[20] = fmaf(float(2), d2, D[20]);
   D[20] += d2;
   D[7] += d1;
   D[16] = fmaf(float(3), dx1d2, D[16]);
   D[30] = fmaf(float(6) * dx1dx1, d3, D[30]);
   D[30] = fmaf(float(2), d2, D[30]);
   D[30] += d2;
   threedxadxb = float(3) * dx1dx0;
   D[13] += dx0d2;
   D[11] += dx1d2;
   D[26] = fmaf(threedxadxb, d3, D[26]);
   D[21] = fmaf(threedxadxb, d3, D[21]);
   D[23] += d2;
   D[23] = fmaf(dx0dx0, d3, D[23]);
   D[23] = fmaf(dx1dx1, d3, D[23]);
   D[9] += d1;
   D[19] = fmaf(float(3), dx2d2, D[19]);
   D[34] = fmaf(float(6) * dx2dx2, d3, D[34]);
   D[34] = fmaf(float(2), d2, D[34]);
   D[34] += d2;
   threedxadxb = float(3) * dx2dx0;
   D[15] += dx0d2;
   D[12] += dx2d2;
   D[29] = fmaf(threedxadxb, d3, D[29]);
   D[22] = fmaf(threedxadxb, d3, D[22]);
   D[25] += d2;
   D[25] = fmaf(dx0dx0, d3, D[25]);
   D[25] = fmaf(dx2dx2, d3, D[25]);
   threedxadxb = float(3) * dx2dx1;
   D[18] += dx1d2;
   D[17] += dx2d2;
   D[33] = fmaf(threedxadxb, d3, D[33]);
   D[31] = fmaf(threedxadxb, d3, D[31]);
   D[32] += d2;
   D[32] = fmaf(dx1dx1, d3, D[32]);
   D[32] = fmaf(dx2dx2, d3, D[32]);
   D[28] = fmaf(dx1dx0, d3, D[28]);
   D[24] = fmaf(dx2dx1, d3, D[24]);
   D[27] = fmaf(dx2dx0, d3, D[27]);
}

template<class T>
CUDA_EXPORT inline void green_direct(expansion &D, const array<T, NDIM> &dX) {
   static const T r0 = 1.0e-9;
// static const T H = options::get().soft_len;
   static const T nthree(-3.0);
   static const T nfive(-5.0);
   static const T nseven(-7.0);
   const T r2 = sqr(dX[0]) + sqr(dX[1]) + sqr(dX[2]);            // 5
   const T r = sqrt(r2);               // 7
   const T rinv = (r > r0) / max(r, r0);  // 3
   const T r2inv = rinv * rinv;        // 1
   const T d0 = -rinv;                 // 1
   const T d1 = -d0 * r2inv;           // 2
   const T d2 = nthree * d1 * r2inv;      // 2
   const T d3 = nfive * d2 * r2inv;    // 2
   const T d4 = nseven * d3 * r2inv;      // 2
   return green_deriv_direct(D, d0, d1, d2, d3, d4, dX);
}

template<class T> // 986 // 251936
CUDA_EXPORT inline void multipole_interaction(expansion &L, const multipole &M, array<T, NDIM> dX, bool do_phi) { // 670/700 + 418 * NT + 50 * NFOUR
   expansion D;
   green_direct(D, dX);
   for (int i = 1 - do_phi; i < LP; i++) {
      L[i] = fmaf(M[0], D[i], L[i]);
   }
   const auto half = (0.5f);
   const auto sixth = (1.0f / 6.0f);
   const auto halfD11 = half * D[11];
   const auto halfD12 = half * D[12];
   const auto halfD13 = half * D[13];
   const auto halfD15 = half * D[15];
   const auto halfD17 = half * D[17];
   const auto halfD18 = half * D[18];                 // 6
   if (do_phi) {
      L[0] = fmaf(M[1], D[4] * half, L[0]);
      L[0] = fmaf(M[7], D[10] * sixth, L[0]);    // 6
      L[0] = fmaf(M[8], halfD11, L[0]);
      L[0] = fmaf(M[9], halfD12, L[0]);
      L[0] = fmaf(M[2], D[5], L[0]);
      L[0] = fmaf(M[10], halfD13, L[0]);
      L[0] = fmaf(M[11], D[14], L[0]);
      L[0] = fmaf(M[3], D[6], L[0]);
      L[0] = fmaf(M[12], halfD15, L[0]);                          // 14
      L[0] = fmaf(M[4], D[7] * half, L[0]);
      L[0] = fmaf(M[13], D[16] * sixth, L[0]);   // 6
      L[0] = fmaf(M[14], halfD17, L[0]);
      L[0] = fmaf(M[5], D[8], L[0]);
      L[0] = fmaf(M[15], halfD18, L[0]);                          // 6
      L[0] = fmaf(M[6], D[9] * half, L[0]);
      L[0] = fmaf(M[16], D[19] * sixth, L[0]);   // 6
   }
   const auto halfD21 = half * D[21];
   const auto halfD22 = half * D[22];
   const auto halfD23 = half * D[23];
   const auto halfD24 = half * D[24];
   const auto halfD25 = half * D[25];
   const auto halfD26 = half * D[26];
   const auto halfD27 = half * D[27];
   const auto halfD28 = half * D[28];
   const auto halfD29 = half * D[29];
   const auto halfD31 = half * D[31];
   const auto halfD32 = half * D[32];
   const auto halfD33 = half * D[33];
   L[1] = fmaf(M[1], D[10] * half, L[1]);
   L[1] = fmaf(M[7], D[20] * sixth, L[1]);
   L[1] = fmaf(M[8], halfD21, L[1]);
   L[1] = fmaf(M[9], halfD22, L[1]);
   L[1] = fmaf(M[2], D[11], L[1]);
   L[1] = fmaf(M[10], halfD23, L[1]);
   L[1] = fmaf(M[11], D[24], L[1]);
   L[1] = fmaf(M[3], D[12], L[1]);
   L[1] = fmaf(M[12], halfD25, L[1]);
   L[1] = fmaf(M[4], halfD13, L[1]);
   L[1] = fmaf(M[13], D[26] * sixth, L[1]);
   L[1] = fmaf(M[14], halfD27, L[1]);
   L[1] = fmaf(M[5], D[14], L[1]);
   L[1] = fmaf(M[15], halfD28, L[1]);
   L[1] = fmaf(M[6], halfD15, L[1]);
   L[1] = fmaf(M[16], D[29] * sixth, L[1]);
   L[2] = fmaf(M[1], halfD11, L[2]);
   L[2] = fmaf(M[7], D[21] * sixth, L[2]);
   L[2] = fmaf(M[8], halfD23, L[2]);
   L[2] = fmaf(M[9], halfD24, L[2]);
   L[2] = fmaf(M[2], D[13], L[2]);
   L[2] = fmaf(M[10], halfD26, L[2]);
   L[2] = fmaf(M[11], D[27], L[2]);
   L[2] = fmaf(M[3], D[14], L[2]);
   L[2] = fmaf(M[12], halfD28, L[2]);
   L[2] = fmaf(M[4], D[16] * half, L[2]);
   L[2] = fmaf(M[13], D[30] * sixth, L[2]);
   L[2] = fmaf(M[14], halfD31, L[2]);
   L[2] = fmaf(M[5], D[17], L[2]);
   L[2] = fmaf(M[15], halfD32, L[2]);
   L[2] = fmaf(M[6], halfD18, L[2]);
   L[2] = fmaf(M[16], D[33] * sixth, L[2]);
   L[3] = fmaf(M[1], halfD12, L[3]);
   L[3] = fmaf(M[7], D[22] * sixth, L[3]);
   L[3] = fmaf(M[8], halfD24, L[3]);
   L[3] = fmaf(M[9], halfD25, L[3]);
   L[3] = fmaf(M[2], D[14], L[3]);
   L[3] = fmaf(M[10], halfD27, L[3]);
   L[3] = fmaf(M[11], D[28], L[3]);
   L[3] = fmaf(M[3], D[15], L[3]);
   L[3] = fmaf(M[12], halfD29, L[3]);
   L[3] = fmaf(M[4], halfD17, L[3]);
   L[3] = fmaf(M[13], D[31] * sixth, L[3]);
   L[3] = fmaf(M[14], halfD32, L[3]);
   L[3] = fmaf(M[5], D[18], L[3]);
   L[3] = fmaf(M[15], halfD33, L[3]);
   L[3] = fmaf(M[6], D[19] * half, L[3]);
   L[3] = fmaf(M[16], D[34] * sixth, L[3]);
   L[4] = fmaf(M[1], D[20] * half, L[4]);
   L[4] = fmaf(M[2], D[21], L[4]);
   L[4] = fmaf(M[3], D[22], L[4]);
   L[4] = fmaf(M[4], halfD23, L[4]);
   L[4] = fmaf(M[5], D[24], L[4]);
   L[4] = fmaf(M[6], halfD25, L[4]);
   L[5] = fmaf(M[1], halfD21, L[5]);
   L[5] = fmaf(M[2], D[23], L[5]);
   L[5] = fmaf(M[3], D[24], L[5]);
   L[5] = fmaf(M[4], halfD26, L[5]);
   L[5] = fmaf(M[5], D[27], L[5]);
   L[5] = fmaf(M[6], halfD28, L[5]);
   L[6] = fmaf(M[1], halfD22, L[6]);
   L[6] = fmaf(M[2], D[24], L[6]);
   L[6] = fmaf(M[3], D[25], L[6]);
   L[6] = fmaf(M[4], halfD27, L[6]);
   L[6] = fmaf(M[5], D[28], L[6]);
   L[6] = fmaf(M[6], halfD29, L[6]);
   L[7] = fmaf(M[1], halfD23, L[7]);
   L[7] = fmaf(M[2], D[26], L[7]);
   L[7] = fmaf(M[3], D[27], L[7]);
   L[7] = fmaf(M[4], D[30] * half, L[7]);
   L[7] = fmaf(M[5], D[31], L[7]);
   L[7] = fmaf(M[6], halfD32, L[7]);
   L[8] = fmaf(M[1], halfD24, L[8]);
   L[8] = fmaf(M[2], D[27], L[8]);
   L[8] = fmaf(M[3], D[28], L[8]);
   L[8] = fmaf(M[4], halfD31, L[8]);
   L[8] = fmaf(M[5], D[32], L[8]);
   L[8] = fmaf(M[6], halfD33, L[8]);
   L[9] = fmaf(M[1], halfD25, L[9]);
   L[9] = fmaf(M[2], D[28], L[9]);
   L[9] = fmaf(M[3], D[29], L[9]);
   L[9] = fmaf(M[4], halfD32, L[9]);
   L[9] = fmaf(M[5], D[33], L[9]);
   L[9] = fmaf(M[6], D[34] * half, L[9]);
}

template<class T> // 516 / 251466
CUDA_EXPORT inline void multipole_interaction(array<exp_real, NDIM + 1> L, const multipole &M, array<T, NDIM> dX,
      bool do_phi) { // 517 / 47428
   expansion D;
   green_direct(D, dX);
    for (int i = 1 - do_phi; i < NDIM + 1; i++) {
      L[i] = M[0] * D[i];
   }
   static const auto half = T(0.5);
   static const auto sixth = T(1.0 / 6.0);
   const auto halfD11 = half * D[11];
   const auto halfD12 = half * D[12];
   const auto halfD13 = half * D[13];
   const auto halfD15 = half * D[15];
   const auto halfD17 = half * D[17];
   const auto halfD18 = half * D[18];
   if (do_phi) {
      L[0] = fma(M[1], D[4] * half, L[0]);
      L[0] = fma(M[7], D[10] * sixth, L[0]);
      L[0] = fma(M[8], halfD11, L[0]);
      L[0] = fma(M[9], halfD12, L[0]);
      L[0] = fma(M[2], D[5], L[0]);
      L[0] = fma(M[10], halfD13, L[0]);
      L[0] = fma(M[11], D[14], L[0]);
      L[0] = fma(M[3], D[6], L[0]);
      L[0] = fma(M[12], halfD15, L[0]);
      L[0] = fma(M[4], D[7] * half, L[0]);
      L[0] = fma(M[13], D[16] * sixth, L[0]);
      L[0] = fma(M[14], halfD17, L[0]);
      L[0] = fma(M[5], D[8], L[0]);
      L[0] = fma(M[15], halfD18, L[0]);
      L[0] = fma(M[6], D[9] * half, L[0]);
      L[0] = fma(M[16], D[19] * sixth, L[0]);
   }
   const auto halfD21 = half * D[21];
   const auto halfD22 = half * D[22];
   const auto halfD23 = half * D[23];
   const auto halfD24 = half * D[24];
   const auto halfD25 = half * D[25];
   const auto halfD26 = half * D[26];
   const auto halfD27 = half * D[27];
   const auto halfD28 = half * D[28];
   const auto halfD29 = half * D[29];
   const auto halfD31 = half * D[31];
   const auto halfD32 = half * D[32];
   const auto halfD33 = half * D[33];
   L[1] = fma(M[1], D[10] * half, L[1]);
   L[1] = fma(M[7], D[20] * sixth, L[1]);
   L[1] = fma(M[8], halfD21, L[1]);
   L[1] = fma(M[9], halfD22, L[1]);
   L[1] = fma(M[2], D[11], L[1]);
   L[1] = fma(M[10], halfD23, L[1]);
   L[1] = fma(M[11], D[24], L[1]);
   L[1] = fma(M[3], D[12], L[1]);
   L[1] = fma(M[12], halfD25, L[1]);
   L[1] = fma(M[4], halfD13, L[1]);
   L[1] = fma(M[13], D[26] * sixth, L[1]);
   L[1] = fma(M[14], halfD27, L[1]);
   L[1] = fma(M[5], D[14], L[1]);
   L[1] = fma(M[15], halfD28, L[1]);
   L[1] = fma(M[6], halfD15, L[1]);
   L[1] = fma(M[16], D[29] * sixth, L[1]);
   L[2] = fma(M[1], halfD11, L[2]);
   L[2] = fma(M[7], D[21] * sixth, L[2]);
   L[2] = fma(M[8], halfD23, L[2]);
   L[2] = fma(M[9], halfD24, L[2]);
   L[2] = fma(M[2], D[13], L[2]);
   L[2] = fma(M[10], halfD26, L[2]);
   L[2] = fma(M[11], D[27], L[2]);
   L[2] = fma(M[3], D[14], L[2]);
   L[2] = fma(M[12], halfD28, L[2]);
   L[2] = fma(M[4], D[16] * half, L[2]);
   L[2] = fma(M[13], D[30] * sixth, L[2]);
   L[2] = fma(M[14], halfD31, L[2]);
   L[2] = fma(M[5], D[17], L[2]);
   L[2] = fma(M[15], halfD32, L[2]);
   L[2] = fma(M[6], halfD18, L[2]);
   L[2] = fma(M[16], D[33] * sixth, L[2]);
   L[3] = fma(M[1], halfD12, L[3]);
   L[3] = fma(M[7], D[22] * sixth, L[3]);
   L[3] = fma(M[8], halfD24, L[3]);
   L[3] = fma(M[9], halfD25, L[3]);
   L[3] = fma(M[2], D[14], L[3]);
   L[3] = fma(M[10], halfD27, L[3]);
   L[3] = fma(M[11], D[28], L[3]);
   L[3] = fma(M[3], D[15], L[3]);
   L[3] = fma(M[12], halfD29, L[3]);
   L[3] = fma(M[4], halfD17, L[3]);
   L[3] = fma(M[13], D[31] * sixth, L[3]);
   L[3] = fma(M[14], halfD32, L[3]);
   L[3] = fma(M[5], D[18], L[3]);
   L[3] = fma(M[15], halfD33, L[3]);
   L[3] = fma(M[6], D[19] * half, L[3]);
   L[3] = fma(M[16], D[34] * sixth, L[3]);
 }

template<class T> // 401 / 251351
CUDA_EXPORT inline void multipole_interaction(expansion &L, const T &M, array<T, NDIM> dX, bool do_phi) { // 390 / 47301
   expansion D;
   green_direct(D, dX);
   for (int i = 0; i < LP; i++) {
      L[i] = fmaf(M, D[i], L[i]);
   }
}

#endif /* COSMICTIGER_INTERACTIONS_HPP_ */
