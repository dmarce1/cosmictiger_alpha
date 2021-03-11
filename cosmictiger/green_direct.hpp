#pragma once


#include <cosmictiger/cuda.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/expansion.hpp>

template<class T>
CUDA_EXPORT inline int green_direct(expansion<T> &D, const array<T, NDIM> &dX, T rmin = 0.f) {
	const T nthree(-3.0f);
	const T nfive(-5.0f);
	const T nseven(-7.0f);
	const T r2 = FMAX(FMA(dX[0], dX[0], FMA(dX[1], dX[1], sqr(dX[2]))), rmin * rmin);            // 5
	const T rinv = RSQRT(r2);                  // FLOP_RSQRT + 3
	const T r2inv = rinv * rinv;        // 1
	const T d0 = -rinv;                 // 1
	const T d1 = -d0 * r2inv;           // 2
	const T d2 = nthree * d1 * r2inv;      // 2
	const T d3 = nfive * d2 * r2inv;    // 2
	const T d4 = nseven * d3 * r2inv;      // 2
	return 18 + FLOP_RSQRT + green_deriv_direct(D, d0, d1, d2, d3, d4, dX);
}


template<class T>
CUDA_EXPORT int inline green_deriv_direct(expansion<T> &D, const T &d0, const T &d1, const T &d2, const T &d3,
		const T &d4, const array<T, NDIM> &dx) {
	T threedxadxb;
	T dxadxbdxc;
	const auto dx0dx0 = dx[0] * dx[0]; // 1
	const auto dx0dx1 = dx[0] * dx[1]; // 1
	const auto dx0dx2 = dx[0] * dx[2]; // 1
	const auto dx1dx1 = dx[1] * dx[1]; // 1
	const auto dx1dx2 = dx[1] * dx[2]; // 1
	const auto dx2dx2 = dx[2] * dx[2]; // 1
	const auto &dx1dx0 = dx0dx1;
	const auto &dx2dx0 = dx0dx2;
	const auto &dx2dx1 = dx1dx2;
	D[0] = d0;
	D[1] = dx[0] * d1;                 // 1
	D[4] = dx0dx0 * d2;                // 1
	dxadxbdxc = dx0dx0 * dx[0];        // 1
	D[10] = dxadxbdxc * d3;            // 1
	D[20] = dxadxbdxc * dx[0] * d4;    // 2
	D[2] = dx[1] * d1;                 // 1
	D[5] = dx1dx0 * d2;                // 1
	dxadxbdxc = dx1dx0 * dx[0];        // 1
	D[11] = dxadxbdxc * d3;            // 1
	D[21] = dxadxbdxc * dx[0] * d4;    // 2
	D[7] = dx1dx1 * d2;                // 1
	dxadxbdxc = dx1dx1 * dx[0];        // 1
	D[13] = dxadxbdxc * d3;            // 1
	D[23] = dxadxbdxc * dx[0] * d4;    // 2
	dxadxbdxc = dx1dx1 * dx[1];        // 1
	D[16] = dxadxbdxc * d3;            // 1
	D[26] = dxadxbdxc * dx[0] * d4;    // 2
	D[30] = dxadxbdxc * dx[1] * d4;    // 2
	D[3] = dx[2] * d1;                 // 1
	D[6] = dx2dx0 * d2;                // 1
	dxadxbdxc = dx2dx0 * dx[0];        // 1
	D[12] = dxadxbdxc * d3;            // 1
	D[22] = dxadxbdxc * dx[0] * d4;    // 2
	D[8] = dx2dx1 * d2;                // 1
	dxadxbdxc = dx2dx1 * dx[0];        // 1
	D[14] = dxadxbdxc * d3;            // 1
	D[24] = dxadxbdxc * dx[0] * d4;    // 2
	dxadxbdxc = dx2dx1 * dx[1];        // 1
	D[17] = dxadxbdxc * d3;            // 1
	D[27] = dxadxbdxc * dx[0] * d4;    // 2
	D[31] = dxadxbdxc * dx[1] * d4;    // 2
	D[9] = dx2dx2 * d2;                // 1
	dxadxbdxc = dx2dx2 * dx[0];        // 1
	D[15] = dxadxbdxc * d3;            // 1
	D[25] = dxadxbdxc * dx[0] * d4;    // 2
	dxadxbdxc = dx2dx2 * dx[1];        // 1
	D[18] = dxadxbdxc * d3;            // 1
	D[28] = dxadxbdxc * dx[0] * d4;    // 2
	D[32] = dxadxbdxc * dx[1] * d4;    // 2
	dxadxbdxc = dx2dx2 * dx[2];        // 1
	D[19] = dxadxbdxc * d3;            // 1
	D[29] = dxadxbdxc * dx[0] * d4;    // 2
	D[33] = dxadxbdxc * dx[1] * d4;    // 2
	D[34] = dxadxbdxc * dx[2] * d4;    // 2

	const auto dx0d2 = dx[0] * d2;          // 1
	const auto dx1d2 = dx[1] * d2;          // 1
	const auto dx2d2 = dx[2] * d2;          // 1
	D[4] += d1;                             // 1
	D[10] = FMA(T(3), dx0d2, D[10]);       // 2
	D[20] = FMA(T(6) * dx0dx0, d3, D[20]); // 3
	D[20] = FMA(T(2), d2, D[20]);          // 2
	D[20] += d2;                            // 1
	D[7] += d1;                             // 1
	D[16] = FMA(T(3), dx1d2, D[16]);       // 2
	D[30] = FMA(T(6) * dx1dx1, d3, D[30]); // 3
	D[30] = FMA(T(2), d2, D[30]);          // 2
	D[30] += d2;                            // 1
	threedxadxb = T(3) * dx1dx0;            // 1
	D[13] += dx0d2;                         // 1
	D[11] += dx1d2;                         // 1
	D[26] = FMA(threedxadxb, d3, D[26]);   // 2
	D[21] = FMA(threedxadxb, d3, D[21]);   // 2
	D[23] += d2;                            // 1
	D[23] = FMA(dx0dx0, d3, D[23]);        // 2
	D[23] = FMA(dx1dx1, d3, D[23]);        // 2
	D[9] += d1;                             // 1
	D[19] = FMA(T(3), dx2d2, D[19]);       // 2
	D[34] = FMA(T(6) * dx2dx2, d3, D[34]); // 2
	D[34] = FMA(T(2), d2, D[34]);          // 2
	D[34] += d2;                            // 1
	threedxadxb = T(3) * dx2dx0;            // 1
	D[15] += dx0d2;                         // 1
	D[12] += dx2d2;                         // 1
	D[29] = FMA(threedxadxb, d3, D[29]);   // 2
	D[22] = FMA(threedxadxb, d3, D[22]);   // 2
	D[25] += d2;                            // 1
	D[25] = FMA(dx0dx0, d3, D[25]);        // 2
	D[25] = FMA(dx2dx2, d3, D[25]);        // 2
	threedxadxb = T(3) * dx2dx1;            // 1
	D[18] += dx1d2;                         // 1
	D[17] += dx2d2;                         // 1
	D[33] = FMA(threedxadxb, d3, D[33]);   // 2
	D[31] = FMA(threedxadxb, d3, D[31]);   // 2
	D[32] += d2;                            // 1
	D[32] = FMA(dx1dx1, d3, D[32]);        // 2
	D[32] = FMA(dx2dx2, d3, D[32]);        // 2
	D[28] = FMA(dx1dx0, d3, D[28]);        // 2
	D[24] = FMA(dx2dx1, d3, D[24]);        // 2
	D[27] = FMA(dx2dx0, d3, D[27]);        // 2
	return 135;
}
