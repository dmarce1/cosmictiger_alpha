/*
 * defs.hpp
 *
 *  Created on: Jan 23, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_DEFS_HPP_
#define COSMICTIGER_DEFS_HPP_

#include <cstdio>
#include <cstdlib>

//#define PERIODIC_OFF

//#define NO_DIPOLE

#define ORDER 6
#define LORDER ORDER
#define MORDER (LORDER-1)

#define EWALD_ALPHA 2.0

template<class ...Args>
#ifdef __CUDA_ARCH__
__device__
#endif
inline void print(const char* fmt, Args ...args) {
	printf(fmt, args...);
#ifndef __CUDA_ARCH__
	fflush(stdout);
#endif
}

#define PRINT(...) print(__VA_ARGS__)

#define NDIM 3
#define FULL_MASK 0xFFFFFFFF

#define NNEIGHBORS 32
#define SPH_TOLER 1.0e-6

//#define TEST_CHECKLIST_TIME

//#define TIMINGS
//#define PERIODIC_OFF

//#define PERIODIC_OFF

//#define TEST_TREE
#ifndef NDEBUG
#define TEST_RADIX
#define TEST_BOUNDS
#endif

#define WARP_SIZE 32

//#define USE_READMOSTLY
//#define USE_NAN_TEST
#define TEST_FORCE
//#define TEST_STACK
#define CONFORMAL_TIME
#define N_TEST_PARTS (100)

#define NKICKS 10

#define KICK_PP_MAX (10*32)
#define MAX_BUCKET_SIZE 160

#define MIN_ACTIVE_PER_BLOCK 16
#define MAX_RUNG 24
#define TREE_MAX_DEPTH 54
#define TREE_RADIX_CUSHION -9
#define TREE_RADIX_MIN 3
#define TREE_RADIX_MAX 54
#define TREE_SORT_MULTITHREAD
#define TREE_MIN_PARTS2THREAD (64*1024)
#define EWALD_BLOCK_SIZE 128
#define KICK_BLOCK_SIZE 32
#define KICK_PC_MAX (1*32)
#define GROUP_BUCKET_SIZE 256
#define GROUP_SIZE MAX_BUCKET_SIZE
#define GPU_QUEUE_SIZE (1024*1024)
#define KICK_GRID_SIZE (256)
//#define MIN_GPU_PARTS (8*1024)

#define MIN_DX (0.00000001f)
#define MIN_RUNG 1

#define DRIFT_BLOCK_SIZE 1024

#define FLOP_RSQRT 4
#define FLOP_DIV 5
#define FLOP_SQRT 5
#define FLOP_EXP 5
#define FLOP_SINCOS 8

//#define HIPRECISION

#define SINK_BIAS 1.5f

#define MIN_PC_PARTS 38
#define MIN_CP_PARTS 19

#define NCHILD 2
//#define PARALLEL_RADIX

#define ALLOCATION_PAGE_SIZE (1024*1024LL)
#define MIN_CUDA_SORT 65536LL
#define OVERSUBSCRIPTION 8


#define COUNT_FLOPS

#ifdef TEST_BOUNDS
#define BOUNDS_CHECK1(a,b,c)                                                              \
		if( a < b || a >= c ) {                                                             \
			PRINT( "Bounds check failed on line %i in file %s:\n", __LINE__, __FILE__);     \
			PRINT( "%li is not between %li and %li\n", (size_t) a, (size_t) b, (size_t) c); \
			assert(false);                                                                         \
		}
#define BOUNDS_CHECK2(a,b)                                                            \
		if( a >= b ) {                                                                  \
			PRINT( "Bounds check failed on line %i in file %s:\n", __LINE__, __FILE__); \
			PRINT( "%li should be less than %li\n",(size_t)  a, (size_t) b);            \
			assert(false);                                                                     \
		}
#else
#define BOUNDS_CHECK1(a,b,c)
#define BOUNDS_CHECK2(a,b)
#endif

#ifdef __CUDA_ARCH__
#define ABORT() __trap()
#else
#define ABORT() abort()
#endif

#define EWALD_MIN_DIST (0.25f)
#define EWALD_MIN_DIST2 (EWALD_MIN_DIST *EWALD_MIN_DIST)
#define EWALD_REAL_CUTOFF2 (2.6f*2.6f)

template<class A, class B>
struct pair {
	A first;
	B second;
	template<class Arc>
	void serialize(Arc&& arc, unsigned) {
		arc & first;
		arc & second;
	}
};

#ifdef __CUDA_ARCH__
#define LDG(a) __ldg(a)
#else
#define LDG(a) (*(a))
#endif

#define FREAD(a,b,c,d) __safe_fread(a,b,c,d,__LINE__,__FILE__)

static void __safe_fread(void* src, size_t size, size_t count, FILE* fp, int line, const char* file) {
	auto read = fread(src, size, count, fp);
	if (read != count) {
		PRINT("Attempt to read %li elements of size %li in %s on line %i failed - only %li elements read.\n", count, size,
				file, line, read);
		abort();
	}
}

#define ERROR() \
	PRINT( "ERROR in %s line %i from rank %i\n", __FILE__, __LINE__, hpx_rank()); \
	abort()

#endif /* COSMICTIGER_DEFS_HPP_ */
