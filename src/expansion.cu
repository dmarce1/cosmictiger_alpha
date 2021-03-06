/*
 * expansion.cpp
 *
 *  Created on: Feb 10, 2021
 *      Author: dmarce1
 */

#include <cosmictiger/expansion.hpp>
#include <cosmictiger/array.hpp>




__device__ expansion<float> Lfactor_gpu;
expansion<float> Lfactor_cpu;

__device__ void expansion_init() {
   for (int i = 0; i < LP; i++) {
      Lfactor_gpu[i] = float(0.0);
   }
   Lfactor_gpu() += float(1);
   for (int a = 0; a < NDIM; ++a) {
      Lfactor_gpu(a) += float(1.0);
      for (int b = 0; b < NDIM; ++b) {
         Lfactor_gpu(a, b) += float(0.5);
         for (int c = 0; c < NDIM; ++c) {
            Lfactor_gpu(a, b, c) += float(1.0 / 6.0);
            for (int d = 0; d < NDIM; ++d) {
               Lfactor_gpu(a, b, c, d) += float(1.0 / 24.0);
            }
         }
      }
   }
}

__host__ void expansion_init_cpu() {
   for (int i = 0; i < LP; i++) {
      Lfactor_cpu[i] = float(0.0);
   }
   Lfactor_cpu() += float(1);
   for (int a = 0; a < NDIM; ++a) {
      Lfactor_cpu(a) += float(1.0);
      for (int b = 0; b < NDIM; ++b) {
         Lfactor_cpu(a, b) += float(0.5);
         for (int c = 0; c < NDIM; ++c) {
            Lfactor_cpu(a, b, c) += float(1.0 / 6.0);
            for (int d = 0; d < NDIM; ++d) {
               Lfactor_cpu(a, b, c, d) += float(1.0 / 24.0);
            }
         }
      }
   }
}

CUDA_EXPORT expansion<float>& shift_expansion(expansion<float> &me,
      const array<float, NDIM> &dX) {
#ifdef __CUDA_ARCH__
	const auto& Lfactor = Lfactor_gpu;
#else
	const auto& Lfactor = Lfactor_cpu;
#endif
	for (int a = 0; a < 3; a++) {
      me() += me(a) * dX[a];
      for (int b = 0; b <= a; b++) {
         me() += me(a, b) * dX[a] * dX[b] * Lfactor(a, b);
         for (int c = 0; c <= b; c++) {
            me() += me(a, b, c) * dX[a] * dX[b] * dX[c] * Lfactor(a, b, c);
            for (int d = 0; d <= c; d++) {
               me() += me(a, b, c, d) * dX[a] * dX[b] * dX[c] * dX[d] * Lfactor(a, b, c, d);
            }
         }
      }
   }
   for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) {
         me(a) += me(a, b) * dX[b];
         for (int c = 0; c <= b; c++) {
            me(a) += me(a, b, c) * dX[b] * dX[c] * Lfactor(b, c);
            for (int d = 0; d <= c; d++) {
               me(a) += me(a, b, c, d) * dX[b] * dX[c] * dX[d] * Lfactor(b, c, d);
            }
         }
      }
   }
   for (int a = 0; a < 3; a++) {
      for (int b = 0; b <= a; b++) {
         for (int c = 0; c < NDIM; c++) {
            me(a, b) += me(a, b, c) * dX[c];
            for (int d = 0; d <= c; d++) {
               me(a, b) += me(a, b, c, d) * dX[c] * dX[d] * Lfactor(c, d);
            }
         }
      }
   }

   for (int a = 0; a < 3; a++) {
      for (int b = 0; b <= a; b++) {
         for (int c = 0; c <= b; c++) {
            for (int d = 0; d < 3; d++) {
               me(a, b, c) += me(a, b, c, d) * dX[d];
            }
         }
      }
   }

   return me;
}

CUDA_EXPORT void shift_expansion(expansion<float> &me, array<float, NDIM> &g, float &phi,
      const array<float, NDIM> &dX) {
#ifdef __CUDA_ARCH__
	const auto& Lfactor = Lfactor_gpu;
#else
	const auto& Lfactor = Lfactor_cpu;
#endif
   phi = me();
   for (int a = 0; a < 3; a++) {
      phi += me(a) * dX[a];
      for (int b = a; b < 3; b++) {
         phi += me(a, b) * dX[a] * dX[b] * Lfactor(a, b);
         for (int c = b; c < 3; c++) {
            phi += me(a, b, c) * dX[a] * dX[b] * dX[c] * Lfactor(a, b, c);
            for (int d = c; d < 3; d++) {
               phi += me(a, b, c, d) * dX[a] * dX[b] * dX[c] * dX[d] * Lfactor(a, b, c, d);
            }
         }
      }
   }
   for (int a = 0; a < 3; a++) {
      g[a] = -me(a);
      for (int b = 0; b < 3; b++) {
         g[a] -= me(a, b) * dX[b];
         for (int c = b; c < 3; c++) {
            g[a] -= me(a, b, c) * dX[b] * dX[c] * Lfactor(b, c);
            for (int d = c; d < 3; d++) {
               g[a] -= me(a, b, c, d) * dX[b] * dX[c] * dX[d] * Lfactor(b, c, d);
            }
         }
      }
   }
}
