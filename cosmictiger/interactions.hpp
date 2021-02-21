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

//#define EWALD_DOUBLE_PRECISION


template<class T>
CUDA_EXPORT int inline green_deriv_direct(expansion<T> &D, const T &d0, const T &d1, const T &d2, const T &d3,
      const T &d4, const array<T, NDIM> &dx);

struct ewald_indices {
private:
   array<hifloat, NDIM> *h;
   size_t count;
public:
   CUDA_EXPORT
   size_t size() const {
      return count;
   }
   CUDA_EXPORT
   array<hifloat, NDIM> get(size_t i) const {
      assert(i < count);
      return h[i];
   }
   ~ewald_indices() {
      CUDA_FREE(h);
   }
   ewald_indices(int n2max, bool nozero) {
      const int nmax = sqrt(n2max) + 1;
      CUDA_MALLOC(h, (2 * nmax + 1) * (2 * nmax + 1) * (2 * nmax + 1));
      array<hifloat, NDIM> this_h;
      count = 0;
      for (int i = -nmax; i <= nmax; i++) {
         for (int j = -nmax; j <= nmax; j++) {
            for (int k = -nmax; k <= nmax; k++) {
               if (i * i + j * j + k * k <= n2max) {
                  this_h[0] = i;
                  this_h[1] = j;
                  this_h[2] = k;
                  const auto hdot = sqr(this_h[0]) + sqr(this_h[1]) + sqr(this_h[2]);
                  if (!nozero || hdot > 0) {
                     h[count++] = this_h;
                  }
               }
            }
         }
      }
   }
};

struct periodic_parts {
private:
   expansion<hifloat> *L;
   size_t count;
public:
   ~periodic_parts() {
      CUDA_FREE(L);
   }
   periodic_parts() {
      const ewald_indices indices(EWALD_NFOUR, true);
      CUDA_MALLOC(L, indices.size());
      count = 0;
      for (int i = 0; i < indices.size(); i++) {
         array<hifloat, NDIM> h = indices.get(i);
         const hifloat h2 = sqr(h[0]) + sqr(h[1]) + sqr(h[2]);                     // 5 OP
         expansion<hifloat> D;
         D = 0.0;
         if (h2 > 0) {
            const hifloat c0 = 1.0 / h2 * exp(-M_PI * M_PI * h2 / 4.0);
            D() = -(1.0 / M_PI) * c0;
            for (int a = 0; a < NDIM; a++) {
               D(a) = 2.0 * h[a] * c0;
               for (int b = 0; b <= a; b++) {
                  D(a, b) = 4.0 * M_PI * h[a] * h[b] * c0;
                  for (int c = 0; c <= b; c++) {
                     D(a, b, c) = -8.0 * M_PI * M_PI * h[a] * h[b] * h[c] * c0;
                     for (int d = 0; d <= c; d++) {
                        D(a, b, c, d) = -16.0 * M_PI * M_PI * M_PI * h[a] * h[b] * h[c] * h[d] * c0;
                     }
                  }
               }

            }
            L[count++] = D;
         }
      }
   }
   CUDA_EXPORT
   size_t size() const {
      return count;
   }
   CUDA_EXPORT
   expansion<hifloat> get(size_t i) const {
      assert(i < count);
      return L[i];
   }
};

#ifndef TREECU
extern CUDA_DEVICE ewald_indices *four_indices_ptr;
extern CUDA_DEVICE ewald_indices *real_indices_ptr;
extern CUDA_DEVICE periodic_parts *periodic_parts_ptr;
#endif

#ifdef HIPRECISION
#define SINCOS sincos
#define EXP exp
#define ERFC erfc
#else
#define SINCOS sincosf
#define EXP expf
#define ERFC erfcf
#endif

template<class T>
CUDA_EXPORT inline int green_direct(expansion<T> &D, const array<T, NDIM> &dX) {
   static const T r0 = 1.0e-9;
// static const T H = options::get().soft_len;
   static const T nthree(-3.0);
   static const T nfive(-5.0);
   static const T nseven(-7.0);
   const T r2 = sqr(dX[0]) + sqr(dX[1]) + sqr(dX[2]);            // 5
   const T r = sqrt(r2);               // 7
   const T rinv = (r > r0) / fmax(r, r0);  // 3
   const T r2inv = rinv * rinv;        // 1
   const T d0 = -rinv;                 // 1
   const T d1 = -d0 * r2inv;           // 2
   const T d2 = nthree * d1 * r2inv;      // 2
   const T d3 = nfive * d2 * r2inv;    // 2
   const T d4 = nseven * d3 * r2inv;      // 2
   return 25 + green_deriv_direct(D, d0, d1, d2, d3, d4, dX);
}

CUDA_DEVICE inline int green_deriv_ewald(expansion<hifloat> &D, const hifloat &d0, const hifloat &d1,
      const hifloat &d2, const hifloat &d3, const hifloat &d4, const array<hifloat, NDIM> &dx) {
   hifloat threedxadxb;
   hifloat dxadxbdxc;
   const auto dx0dx0 = dx[0] * dx[0];
   const auto dx0dx1 = dx[0] * dx[1];
   const auto dx0dx2 = dx[0] * dx[2];
   const auto dx1dx1 = dx[1] * dx[1];
   const auto dx1dx2 = dx[1] * dx[2];
   const auto dx2dx2 = dx[2] * dx[2];
   const auto &dx1dx0 = dx0dx1;
   const auto &dx2dx0 = dx0dx2;
   const auto &dx2dx1 = dx1dx2;
   D[0] += d0;
   D[1] = fma(dx[0], d1, D[1]);
   D[4] = fma(dx0dx0, d2, D[4]);
   dxadxbdxc = dx0dx0 * dx[0];
   D[10] = fma(dxadxbdxc, d3, D[10]);
   D[20] = fma(dxadxbdxc * dx[0], d4, D[20]);
   D[2] = fma(dx[1], d1, D[2]);
   D[5] = fma(dx1dx0, d2, D[5]);
   dxadxbdxc = dx1dx0 * dx[0];
   D[11] = fma(dxadxbdxc, d3, D[11]);
   D[21] = fma(dxadxbdxc * dx[0], d4, D[21]);
   D[7] = fma(dx1dx1, d2, D[7]);
   dxadxbdxc = dx1dx1 * dx[0];
   D[13] = fma(dxadxbdxc, d3, D[13]);
   D[23] = fma(dxadxbdxc * dx[0], d4, D[23]);
   dxadxbdxc = dx1dx1 * dx[1];
   D[16] = fma(dxadxbdxc, d3, D[16]);
   D[26] = fma(dxadxbdxc * dx[0], d4, D[26]);
   D[30] = fma(dxadxbdxc * dx[1], d4, D[30]);
   D[3] = fma(dx[2], d1, D[3]);
   D[6] = fma(dx2dx0, d2, D[6]);
   dxadxbdxc = dx2dx0 * dx[0];
   D[12] = fma(dxadxbdxc, d3, D[12]);
   D[22] = fma(dxadxbdxc * dx[0], d4, D[22]);
   D[8] = fma(dx2dx1, d2, D[8]);
   dxadxbdxc = dx2dx1 * dx[0];
   D[14] = fma(dxadxbdxc, d3, D[14]);
   D[24] = fma(dxadxbdxc * dx[0], d4, D[24]);
   dxadxbdxc = dx2dx1 * dx[1];
   D[17] = fma(dxadxbdxc, d3, D[17]);
   D[27] = fma(dxadxbdxc * dx[0], d4, D[27]);
   D[31] = fma(dxadxbdxc * dx[1], d4, D[31]);
   D[9] = fma(dx2dx2, d2, D[9]);
   dxadxbdxc = dx2dx2 * dx[0];
   D[15] = fma(dxadxbdxc, d3, D[15]);
   D[25] = fma(dxadxbdxc * dx[0], d4, D[25]);
   dxadxbdxc = dx2dx2 * dx[1];
   D[18] = fma(dxadxbdxc, d3, D[18]);
   D[28] = fma(dxadxbdxc * dx[0], d4, D[28]);
   D[32] = fma(dxadxbdxc * dx[1], d4, D[32]);
   dxadxbdxc = dx2dx2 * dx[2];
   D[19] = fma(dxadxbdxc, d3, D[19]);
   D[29] = fma(dxadxbdxc * dx[0], d4, D[29]);
   D[33] = fma(dxadxbdxc * dx[1], d4, D[33]);
   D[34] = fma(dxadxbdxc * dx[2], d4, D[34]);

   const auto dx0d2 = dx[0] * d2;
   const auto dx1d2 = dx[1] * d2;
   const auto dx2d2 = dx[2] * d2;
   D[4] += d1;
   D[10] = fma(hifloat(3), dx0d2, D[10]);
   D[20] = fma(hifloat(6) * dx0dx0, d3, D[20]);
   D[20] = fma(hifloat(2), d2, D[20]);
   D[20] += d2;
   D[7] += d1;
   D[16] = fma(hifloat(3), dx1d2, D[16]);
   D[30] = fma(hifloat(6) * dx1dx1, d3, D[30]);
   D[30] = fma(hifloat(2), d2, D[30]);
   D[30] += d2;
   threedxadxb = hifloat(3) * dx1dx0;
   D[13] += dx0d2;
   D[11] += dx1d2;
   D[26] = fma(threedxadxb, d3, D[26]);
   D[21] = fma(threedxadxb, d3, D[21]);
   D[23] += d2;
   D[23] = fma(dx0dx0, d3, D[23]);
   D[23] = fma(dx1dx1, d3, D[23]);
   D[9] += d1;
   D[19] = fma(hifloat(3), dx2d2, D[19]);
   D[34] = fma(hifloat(6) * dx2dx2, d3, D[34]);
   D[34] = fma(hifloat(2), d2, D[34]);
   D[34] += d2;
   threedxadxb = hifloat(3) * dx2dx0;
   D[15] += dx0d2;
   D[12] += dx2d2;
   D[29] = fma(threedxadxb, d3, D[29]);
   D[22] = fma(threedxadxb, d3, D[22]);
   D[25] += d2;
   D[25] = fma(dx0dx0, d3, D[25]);
   D[25] = fma(dx2dx2, d3, D[25]);
   threedxadxb = hifloat(3) * dx2dx1;
   D[18] += dx1d2;
   D[17] += dx2d2;
   D[33] = fma(threedxadxb, d3, D[33]);
   D[31] = fma(threedxadxb, d3, D[31]);
   D[32] += d2;
   D[32] = fma(dx1dx1, d3, D[32]);
   D[32] = fma(dx2dx2, d3, D[32]);
   D[28] = fma(dx1dx0, d3, D[28]);
   D[24] = fma(dx2dx1, d3, D[24]);
   D[27] = fma(dx2dx0, d3, D[27]);
   return 135;
}

CUDA_DEVICE inline int green_ewald(expansion<hifloat> &D, const array<hifloat, NDIM> &X) {
   const auto &hparts = *periodic_parts_ptr;
   const auto &four_indices = *four_indices_ptr;
   const auto &real_indices = *real_indices_ptr;
   const hifloat fouroversqrtpi(4.0 / sqrt(M_PI));
   static const hifloat one(1.0);
   static const hifloat nthree(-3.0);
   static const hifloat nfour(-4.0);
   static const hifloat nfive(-5.0);
   static const hifloat nseven(-7.0);
   static const hifloat neight(-8.0);
   static const hifloat rcut(1.0e-6);
   const hifloat r = sqrt(sqr(X[0]) + sqrt(X[1]) + sqr(X[2]));                   // 5
   const hifloat zmask = r > rcut;    // 1
   expansion<hifloat> &Dreal = D;
   expansion<hifloat> Dfour;
   int flops = 0;
   Dreal = 0.0;
   Dfour = 0.0;
   for (int i = 0; i < real_indices.size(); i++) {
      const auto n = real_indices.get(i);
      array<hifloat, NDIM> dx;
      for (int dim = 0; dim < NDIM; dim++) {                                        // 6
         dx[dim] = X[dim] - n[dim];
      }
      const hifloat r2 = sqr(dx[0]) + sqr(dx[1]) + sqr(dx[2]);                   // 5
      if (r2 < (EWALD_REAL_CUTOFF * EWALD_REAL_CUTOFF)) {                           // 1
         flops += FLOPS_EWALD;
         const hifloat r = sqrt(r2);                                             // 1
         const hifloat cmask = one - (sqr(n[0]) + sqr(n[1]) + sqr(n[2]) > 0.0);  // 7
         const hifloat mask = (one - (one - zmask) * cmask);                     // 3
         const hifloat rinv = mask / fmaxf(r, rcut);                             // 2
         const hifloat r2inv = rinv * rinv;                                      // 1
         const hifloat r3inv = r2inv * rinv;                                     // 1
         const hifloat exp0 = EXP(nfour * r2);                                   // 26
         const hifloat erfc0 = ERFC(2.f * r);                                    // 10
         const hifloat expfactor = fouroversqrtpi * r * exp0;                    // 2
         const hifloat e1 = expfactor * r3inv;                                   // 1
         const hifloat e2 = neight * e1;                                         // 1
         const hifloat e3 = neight * e2;                                         // 1
         const hifloat e4 = neight * e3;                                         // 1
         const hifloat d0 = -erfc0 * rinv;                                       // 2
         const hifloat d1 = fma(-d0, r2inv, e1);                                 // 3
         const hifloat d2 = fma(nthree * d1, r2inv, e2);                         // 3
         const hifloat d3 = fma(nfive * d2, r2inv, e3);                          // 3
         const hifloat d4 = fma(nseven * d3, r2inv, e4);                         // 3
         green_deriv_ewald(Dreal, d0, d1, d2, d3, d4, dx);
      }

   }
   static const hifloat twopi = 2.0 * M_PI;

   for (int i = 0; i < four_indices.size(); i++) {
      const auto &h = four_indices.get(i);
      const auto &hpart = hparts.get(i);
//    printf( "H = %e %e %e\n", h[0], h[1], h[2]);
      const hifloat h2 = sqrt(h[0]) + sqr(h[1]) + sqr(h[2]); // 5
      const hifloat hdotx = h[0] * X[0] + h[1] * X[1] + h[2] * X[2]; // 5
      hifloat co;
      hifloat so;
      SINCOS(twopi * hdotx, &so, &co);                // 35
      Dfour[0] = fma(hpart[0], co, Dfour[0]);         // 2
      Dfour[1] = fma(hpart[1], so, Dfour[1]);         // 2
      Dfour[2] = fma(hpart[2], so, Dfour[2]);         // 2
      Dfour[3] = fma(hpart[3], so, Dfour[3]);         // 2
      Dfour[4] = fma(hpart[4], co, Dfour[4]);         // 2
      Dfour[5] = fma(hpart[5], co, Dfour[5]);         // 2
      Dfour[6] = fma(hpart[6], co, Dfour[6]);         // 2
      Dfour[7] = fma(hpart[7], co, Dfour[7]);         // 2
      Dfour[8] = fma(hpart[8], co, Dfour[8]);         // 2
      Dfour[9] = fma(hpart[9], co, Dfour[9]);         // 2
      Dfour[10] = fma(hpart[10], so, Dfour[10]);      // 2
      Dfour[11] = fma(hpart[11], so, Dfour[11]);      // 2
      Dfour[12] = fma(hpart[12], so, Dfour[12]);      // 2
      Dfour[13] = fma(hpart[13], so, Dfour[13]);      // 2
      Dfour[14] = fma(hpart[14], so, Dfour[14]);      // 2
      Dfour[15] = fma(hpart[15], so, Dfour[15]);      // 2
      Dfour[16] = fma(hpart[16], so, Dfour[16]);      // 2
      Dfour[17] = fma(hpart[17], so, Dfour[17]);      // 2
      Dfour[18] = fma(hpart[18], so, Dfour[18]);      // 2
      Dfour[19] = fma(hpart[19], so, Dfour[19]);      // 2
      Dfour[20] = fma(hpart[20], co, Dfour[20]);      // 2
      Dfour[21] = fma(hpart[21], co, Dfour[21]);      // 2
      Dfour[22] = fma(hpart[22], co, Dfour[22]);      // 2
      Dfour[23] = fma(hpart[23], co, Dfour[23]);      // 2
      Dfour[24] = fma(hpart[24], co, Dfour[24]);      // 2
      Dfour[25] = fma(hpart[25], co, Dfour[25]);      // 2
      Dfour[26] = fma(hpart[26], co, Dfour[26]);      // 2
      Dfour[27] = fma(hpart[27], co, Dfour[27]);      // 2
      Dfour[28] = fma(hpart[28], co, Dfour[28]);      // 2
      Dfour[30] = fma(hpart[30], co, Dfour[30]);      // 2
      Dfour[29] = fma(hpart[29], co, Dfour[29]);      // 2
      Dfour[31] = fma(hpart[31], co, Dfour[31]);      // 2
      Dfour[32] = fma(hpart[32], co, Dfour[32]);      // 2
      Dfour[33] = fma(hpart[33], co, Dfour[33]);      // 2
      Dfour[34] = fma(hpart[34], co, Dfour[34]);      // 2
   }
   for (int i = 0; i < LP; i++) {                     // 17
      Dreal[i] += Dfour[i];
   }
   expansion<hifloat> D1;
   green_direct(D1, X);
   D() = (M_PI / 4.0) + D();                          // 1
   for (int i = 0; i < LP; i++) {                     // 35
      D[i] = fma(-zmask, D1[i], D[i]);
   }
   return flops;
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
   D[10] = fmaf(T(3), dx0d2, D[10]);       // 2
   D[20] = fmaf(T(6) * dx0dx0, d3, D[20]); // 3
   D[20] = fmaf(T(2), d2, D[20]);          // 2
   D[20] += d2;                            // 1
   D[7] += d1;                             // 1
   D[16] = fmaf(T(3), dx1d2, D[16]);       // 2
   D[30] = fmaf(T(6) * dx1dx1, d3, D[30]); // 3
   D[30] = fmaf(T(2), d2, D[30]);          // 2
   D[30] += d2;                            // 1
   threedxadxb = T(3) * dx1dx0;            // 1
   D[13] += dx0d2;                         // 1
   D[11] += dx1d2;                         // 1
   D[26] = fmaf(threedxadxb, d3, D[26]);   // 2
   D[21] = fmaf(threedxadxb, d3, D[21]);   // 2
   D[23] += d2;                            // 1
   D[23] = fmaf(dx0dx0, d3, D[23]);        // 2
   D[23] = fmaf(dx1dx1, d3, D[23]);        // 2
   D[9] += d1;                             // 1
   D[19] = fmaf(T(3), dx2d2, D[19]);       // 2
   D[34] = fmaf(T(6) * dx2dx2, d3, D[34]); // 2
   D[34] = fmaf(T(2), d2, D[34]);          // 2
   D[34] += d2;                            // 1
   threedxadxb = T(3) * dx2dx0;            // 1
   D[15] += dx0d2;                         // 1
   D[12] += dx2d2;                         // 1
   D[29] = fmaf(threedxadxb, d3, D[29]);   // 2
   D[22] = fmaf(threedxadxb, d3, D[22]);   // 2
   D[25] += d2;                            // 1
   D[25] = fmaf(dx0dx0, d3, D[25]);        // 2
   D[25] = fmaf(dx2dx2, d3, D[25]);        // 2
   threedxadxb = T(3) * dx2dx1;            // 1
   D[18] += dx1d2;                         // 1
   D[17] += dx2d2;                         // 1
   D[33] = fmaf(threedxadxb, d3, D[33]);   // 2
   D[31] = fmaf(threedxadxb, d3, D[31]);   // 2
   D[32] += d2;                            // 1
   D[32] = fmaf(dx1dx1, d3, D[32]);        // 2
   D[32] = fmaf(dx2dx2, d3, D[32]);        // 2
   D[28] = fmaf(dx1dx0, d3, D[28]);        // 2
   D[24] = fmaf(dx2dx1, d3, D[24]);        // 2
   D[27] = fmaf(dx2dx0, d3, D[27]);        // 2
   return 135;
}

// 986 // 251936
template<class T>
CUDA_EXPORT inline int multipole_interaction(expansion<T> &L, const multipole_type<T> &M, array<T, NDIM> dX,
     bool do_phi) { // 670/700 + 418 * NT + 50 * NFOUR
   expansion<T> D;
   int flops = green_direct(D, dX);
   for (int i = 1 - do_phi; i < LP; i++) {
      L[i] = fmaf(M[0], D[i], L[i]);                  // 35
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
      L[0] = fmaf(M[12], halfD15, L[0]);    // 14
      L[0] = fmaf(M[4], D[7] * half, L[0]);
      L[0] = fmaf(M[13], D[16] * sixth, L[0]);    // 6
      L[0] = fmaf(M[14], halfD17, L[0]);
      L[0] = fmaf(M[5], D[8], L[0]);
      L[0] = fmaf(M[15], halfD18, L[0]);    // 6
      L[0] = fmaf(M[6], D[9] * half, L[0]);
      L[0] = fmaf(M[16], D[19] * sixth, L[0]);    // 6
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
   return flops + 269;
}

// 986 // 251936
CUDA_DEVICE inline int multipole_interaction_ewald(expansion<hifloat> &L, const multipole_type<hifloat> &M,
      array<hifloat, NDIM> dX, bool do_phi) { // 670/700 + 418 * NT + 50 * NFOUR
   expansion<hifloat> D;
   int flops = green_ewald(D, dX);
   for (int i = 1 - do_phi; i < LP; i++) {
      L[i] = fmaf(M[0], D[i], L[i]);              // 35
   }
   const auto half = (0.5f);
   const auto sixth = (1.0f / 6.0f);
   const auto halfD11 = half * D[11];             // 1
   const auto halfD12 = half * D[12];             // 1
   const auto halfD13 = half * D[13];             // 1
   const auto halfD15 = half * D[15];             // 1
   const auto halfD17 = half * D[17];             // 1
   const auto halfD18 = half * D[18];             // 1
   if (do_phi) {
      L[0] = fmaf(M[1], D[4] * half, L[0]);       // 2
      L[0] = fmaf(M[7], D[10] * sixth, L[0]);     // 3
      L[0] = fmaf(M[8], halfD11, L[0]);           // 2
      L[0] = fmaf(M[9], halfD12, L[0]);           // 2
      L[0] = fmaf(M[2], D[5], L[0]);              // 2
      L[0] = fmaf(M[10], halfD13, L[0]);          // 2
      L[0] = fmaf(M[11], D[14], L[0]);            // 2
      L[0] = fmaf(M[3], D[6], L[0]);              // 2
      L[0] = fmaf(M[12], halfD15, L[0]);          // 2
      L[0] = fmaf(M[4], D[7] * half, L[0]);       // 3
      L[0] = fmaf(M[13], D[16] * sixth, L[0]);    // 3
      L[0] = fmaf(M[14], halfD17, L[0]);          // 2
      L[0] = fmaf(M[5], D[8], L[0]);              // 2
      L[0] = fmaf(M[15], halfD18, L[0]);          // 2
      L[0] = fmaf(M[6], D[9] * half, L[0]);       // 3
      L[0] = fmaf(M[16], D[19] * sixth, L[0]);    // 3
   }
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
   L[1] = fmaf(M[1], D[10] * half, L[1]);   // 3
   L[1] = fmaf(M[7], D[20] * sixth, L[1]);  // 3
   L[1] = fmaf(M[8], halfD21, L[1]);        // 2
   L[1] = fmaf(M[9], halfD22, L[1]);        // 2
   L[1] = fmaf(M[2], D[11], L[1]);          // 2
   L[1] = fmaf(M[10], halfD23, L[1]);       // 2
   L[1] = fmaf(M[11], D[24], L[1]);         // 2
   L[1] = fmaf(M[3], D[12], L[1]);          // 2
   L[1] = fmaf(M[12], halfD25, L[1]);       // 2
   L[1] = fmaf(M[4], halfD13, L[1]);        // 2
   L[1] = fmaf(M[13], D[26] * sixth, L[1]); // 3
   L[1] = fmaf(M[14], halfD27, L[1]);       // 2
   L[1] = fmaf(M[5], D[14], L[1]);          // 2
   L[1] = fmaf(M[15], halfD28, L[1]);       // 2
   L[1] = fmaf(M[6], halfD15, L[1]);        // 2
   L[1] = fmaf(M[16], D[29] * sixth, L[1]); // 3
   L[2] = fmaf(M[1], halfD11, L[2]);        // 2
   L[2] = fmaf(M[7], D[21] * sixth, L[2]);  // 3
   L[2] = fmaf(M[8], halfD23, L[2]);        // 2
   L[2] = fmaf(M[9], halfD24, L[2]);        // 2
   L[2] = fmaf(M[2], D[13], L[2]);          // 2
   L[2] = fmaf(M[10], halfD26, L[2]);       // 2
   L[2] = fmaf(M[11], D[27], L[2]);         // 2
   L[2] = fmaf(M[3], D[14], L[2]);          // 2
   L[2] = fmaf(M[12], halfD28, L[2]);       // 2
   L[2] = fmaf(M[4], D[16] * half, L[2]);   // 2
   L[2] = fmaf(M[13], D[30] * sixth, L[2]); // 3
   L[2] = fmaf(M[14], halfD31, L[2]);       // 2
   L[2] = fmaf(M[5], D[17], L[2]);          // 2
   L[2] = fmaf(M[15], halfD32, L[2]);       // 2
   L[2] = fmaf(M[6], halfD18, L[2]);        // 2
   L[2] = fmaf(M[16], D[33] * sixth, L[2]); // 3
   L[3] = fmaf(M[1], halfD12, L[3]);        // 2
   L[3] = fmaf(M[7], D[22] * sixth, L[3]);  // 3
   L[3] = fmaf(M[8], halfD24, L[3]);        // 2
   L[3] = fmaf(M[9], halfD25, L[3]);        // 2
   L[3] = fmaf(M[2], D[14], L[3]);          // 2
   L[3] = fmaf(M[10], halfD27, L[3]);       // 2
   L[3] = fmaf(M[11], D[28], L[3]);         // 2
   L[3] = fmaf(M[3], D[15], L[3]);          // 2
   L[3] = fmaf(M[12], halfD29, L[3]);       // 2
   L[3] = fmaf(M[4], halfD17, L[3]);        // 2
   L[3] = fmaf(M[13], D[31] * sixth, L[3]); // 3
   L[3] = fmaf(M[14], halfD32, L[3]);       // 2
   L[3] = fmaf(M[5], D[18], L[3]);          // 2
   L[3] = fmaf(M[15], halfD33, L[3]);       // 2
   L[3] = fmaf(M[6], D[19] * half, L[3]);   // 3
   L[3] = fmaf(M[16], D[34] * sixth, L[3]); // 3
   L[4] = fmaf(M[1], D[20] * half, L[4]);   // 3
   L[4] = fmaf(M[2], D[21], L[4]);          // 2
   L[4] = fmaf(M[3], D[22], L[4]);          // 2
   L[4] = fmaf(M[4], halfD23, L[4]);        // 2
   L[4] = fmaf(M[5], D[24], L[4]);          // 2
   L[4] = fmaf(M[6], halfD25, L[4]);        // 2
   L[5] = fmaf(M[1], halfD21, L[5]);        // 2
   L[5] = fmaf(M[2], D[23], L[5]);          // 2
   L[5] = fmaf(M[3], D[24], L[5]);          // 2
   L[5] = fmaf(M[4], halfD26, L[5]);        // 2
   L[5] = fmaf(M[5], D[27], L[5]);          // 2
   L[5] = fmaf(M[6], halfD28, L[5]);        // 2
   L[6] = fmaf(M[1], halfD22, L[6]);        // 2
   L[6] = fmaf(M[2], D[24], L[6]);          // 2
   L[6] = fmaf(M[3], D[25], L[6]);          // 2
   L[6] = fmaf(M[4], halfD27, L[6]);        // 2
   L[6] = fmaf(M[5], D[28], L[6]);          // 2
   L[6] = fmaf(M[6], halfD29, L[6]);        // 2
   L[7] = fmaf(M[1], halfD23, L[7]);        // 2
   L[7] = fmaf(M[2], D[26], L[7]);          // 2
   L[7] = fmaf(M[3], D[27], L[7]);          // 2
   L[7] = fmaf(M[4], D[30] * half, L[7]);   // 3
   L[7] = fmaf(M[5], D[31], L[7]);          // 2
   L[7] = fmaf(M[6], halfD32, L[7]);        // 2
   L[8] = fmaf(M[1], halfD24, L[8]);        // 2
   L[8] = fmaf(M[2], D[27], L[8]);          // 2
   L[8] = fmaf(M[3], D[28], L[8]);          // 2
   L[8] = fmaf(M[4], halfD31, L[8]);        // 2
   L[8] = fmaf(M[5], D[32], L[8]);          // 2
   L[8] = fmaf(M[6], halfD33, L[8]);        // 2
   L[9] = fmaf(M[1], halfD25, L[9]);        // 2
   L[9] = fmaf(M[2], D[28], L[9]);          // 2
   L[9] = fmaf(M[3], D[29], L[9]);          // 2
   L[9] = fmaf(M[4], halfD32, L[9]);        // 2
   L[9] = fmaf(M[5], D[33], L[9]);          // 2
   L[9] = fmaf(M[6], D[34] * half, L[9]);   // 3
   return flops;
 //  return flops + 269;
}
// 516 / 251466
CUDA_EXPORT inline int multipole_interaction(array<float, NDIM + 1> L, const multipole &M, array<float, NDIM> dX,
      bool do_phi) { // 517 / 47428
   expansion<float> D;
   int flops = green_direct(D, dX);
   for (int i = 1 - do_phi; i < NDIM + 1; i++) {
      L[i] = M[0] * D[i];
   }
   flops += 1 + NDIM;
   static const auto half = float(0.5);
   static const auto sixth = float(1.0 / 6.0);
   const auto halfD11 = half * D[11];             // 1
   const auto halfD12 = half * D[12];             // 1
   const auto halfD13 = half * D[13];             // 1
   const auto halfD15 = half * D[15];             // 1
   const auto halfD17 = half * D[17];             // 1
   const auto halfD18 = half * D[18];             // 1
   if (do_phi) {
      L[0] = fma(M[1], D[4] * half, L[0]);        // 3
      L[0] = fma(M[7], D[10] * sixth, L[0]);      // 3
      L[0] = fma(M[8], halfD11, L[0]);            // 2
      L[0] = fma(M[9], halfD12, L[0]);            // 2
      L[0] = fma(M[2], D[5], L[0]);               // 2
      L[0] = fma(M[10], halfD13, L[0]);           // 2
      L[0] = fma(M[11], D[14], L[0]);             // 2
      L[0] = fma(M[3], D[6], L[0]);               // 2
      L[0] = fma(M[12], halfD15, L[0]);           // 2
      L[0] = fma(M[4], D[7] * half, L[0]);        // 3
      L[0] = fma(M[13], D[16] * sixth, L[0]);     // 3
      L[0] = fma(M[14], halfD17, L[0]);           // 2
      L[0] = fma(M[5], D[8], L[0]);               // 2
      L[0] = fma(M[15], halfD18, L[0]);           // 2
      L[0] = fma(M[6], D[9] * half, L[0]);        // 3
      L[0] = fma(M[16], D[19] * sixth, L[0]);     // 3
   }
   const auto halfD21 = half * D[21];             // 1
   const auto halfD22 = half * D[22];             // 1
   const auto halfD23 = half * D[23];             // 1
   const auto halfD24 = half * D[24];             // 1
   const auto halfD25 = half * D[25];             // 1
   const auto halfD26 = half * D[26];             // 1
   const auto halfD27 = half * D[27];             // 1
   const auto halfD28 = half * D[28];             // 1
   const auto halfD29 = half * D[29];             // 1
   const auto halfD31 = half * D[31];             // 1
   const auto halfD32 = half * D[32];             // 1
   const auto halfD33 = half * D[33];             // 1
   L[1] = fma(M[1], D[10] * half, L[1]);          // 3
   L[1] = fma(M[7], D[20] * sixth, L[1]);         // 3
   L[1] = fma(M[8], halfD21, L[1]);               // 2
   L[1] = fma(M[9], halfD22, L[1]);               // 2
   L[1] = fma(M[2], D[11], L[1]);                 // 2
   L[1] = fma(M[10], halfD23, L[1]);              // 2
   L[1] = fma(M[11], D[24], L[1]);                // 2
   L[1] = fma(M[3], D[12], L[1]);                 // 2
   L[1] = fma(M[12], halfD25, L[1]);              // 2
   L[1] = fma(M[4], halfD13, L[1]);               // 2
   L[1] = fma(M[13], D[26] * sixth, L[1]);        // 3
   L[1] = fma(M[14], halfD27, L[1]);              // 2
   L[1] = fma(M[5], D[14], L[1]);                 // 2
   L[1] = fma(M[15], halfD28, L[1]);              // 2
   L[1] = fma(M[6], halfD15, L[1]);               // 2
   L[1] = fma(M[16], D[29] * sixth, L[1]);        // 3
   L[2] = fma(M[1], halfD11, L[2]);               // 2
   L[2] = fma(M[7], D[21] * sixth, L[2]);         // 3
   L[2] = fma(M[8], halfD23, L[2]);               // 2
   L[2] = fma(M[9], halfD24, L[2]);               // 2
   L[2] = fma(M[2], D[13], L[2]);                 // 2
   L[2] = fma(M[10], halfD26, L[2]);              // 2
   L[2] = fma(M[11], D[27], L[2]);                // 2
   L[2] = fma(M[3], D[14], L[2]);                 // 2
   L[2] = fma(M[12], halfD28, L[2]);              // 2
   L[2] = fma(M[4], D[16] * half, L[2]);          // 3
   L[2] = fma(M[13], D[30] * sixth, L[2]);        // 3
   L[2] = fma(M[14], halfD31, L[2]);              // 2
   L[2] = fma(M[5], D[17], L[2]);                 // 2
   L[2] = fma(M[15], halfD32, L[2]);              // 2
   L[2] = fma(M[6], halfD18, L[2]);               // 2
   L[2] = fma(M[16], D[33] * sixth, L[2]);        // 3
   L[3] = fma(M[1], halfD12, L[3]);               // 2
   L[3] = fma(M[7], D[22] * sixth, L[3]);         // 3
   L[3] = fma(M[8], halfD24, L[3]);               // 2
   L[3] = fma(M[9], halfD25, L[3]);               // 2
   L[3] = fma(M[2], D[14], L[3]);                 // 2
   L[3] = fma(M[10], halfD27, L[3]);              // 2
   L[3] = fma(M[11], D[28], L[3]);                // 2
   L[3] = fma(M[3], D[15], L[3]);                 // 2
   L[3] = fma(M[12], halfD29, L[3]);              // 2
   L[3] = fma(M[4], halfD17, L[3]);               // 2
   L[3] = fma(M[13], D[31] * sixth, L[3]);        // 3
   L[3] = fma(M[14], halfD32, L[3]);              // 2
   L[3] = fma(M[5], D[18], L[3]);                 // 2
   L[3] = fma(M[15], halfD33, L[3]);              // 2
   L[3] = fma(M[6], D[19] * half, L[3]);          // 3
   L[3] = fma(M[16], D[34] * sixth, L[3]);        // 3
   return 159 + flops;
}

template<class T>
CUDA_EXPORT inline int multipole_interaction(expansion<T> &L, const T &M, array<T, NDIM> dX, bool do_phi) { // 390 / 47301
   expansion<T> D;
   green_direct(D, dX);
   for (int i = 0; i < LP; i++) {
      L[i] += M * D[i];
   }
   return LP;
}

#endif /* COSMICTIGER_INTERACTIONS_HPP_ */
