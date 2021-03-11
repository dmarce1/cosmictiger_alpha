/*
 * defs.hpp
 *
 *  Created on: Jan 23, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_DEFS_HPP_
#define COSMICTIGER_DEFS_HPP_

#define NDIM 3
#define FULL_MASK 0xFFFFFFFF

//#define TEST_CHECKLIST_TIME

//#define TIMINGS
//#define PERIODIC_OFF

//#define PERIODIC_OFF

#ifndef NDEBUG
#define TEST_TREE
#define TEST_RADIX
#define TEST_BOUNDS
#endif

#define TEST_FORCE
//#define TEST_STACK
#define N_TEST_PARTS (6*46)

#define MAX_RUNG 64
#define TREE_MAX_DEPTH 54
#define TREE_RADIX_CUSHION -2
#define TREE_RADIX_MIN 3
#define TREE_RADIX_MAX 54
#define TREE_SORT_MULTITHREAD
#define TREE_MIN_PARTS2THREAD (64*1024)
#define KICK_BLOCK_SIZE 32
#define KICK_PP_MAX size_t(16*32)
#define KICK_PC_MAX (2*32)
#define MAX_BUCKET_SIZE 94
#define GROUP_SIZE MAX_BUCKET_SIZE
#define GPU_QUEUE_SIZE (1024*1024)
#define KICK_GRID_SIZE 36
#define KICK_EWALD_GRID_SIZE 1024
#define KICK_OCCUPANCY 8

#define FLOP_RSQRT 4
#define FLOP_DIV 5
#define FLOP_SQRT 5
#define FLOP_EXP 5
#define FLOP_SINCOS 8



//#define HIPRECISION

#define SINK_BIAS 1.5


#define NCHILD 2
//#define PARALLEL_RADIX

#define ALLOCATION_PAGE_SIZE (2*1024*1024LL)
#define MIN_CUDA_SORT 65536LL
#define OVERSUBSCRIPTION 8

#define COUNT_FLOPS

#ifdef TEST_BOUNDS
#define BOUNDS_CHECK1(a,b,c)                                                              \
		if( a < b || a >= c ) {                                                             \
			printf( "Bounds check failed on line %i in file %s:\n", __LINE__, __FILE__);     \
			printf( "%li is not between %li and %li\n", (size_t) a, (size_t) b, (size_t) c); \
			assert(false);                                                                         \
		}
#define BOUNDS_CHECK2(a,b)                                                            \
		if( a >= b ) {                                                                  \
			printf( "Bounds check failed on line %i in file %s:\n", __LINE__, __FILE__); \
			printf( "%li should be less than %li\n",(size_t)  a, (size_t) b);            \
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

#define EWALD_MIN_DIST2 (0.25f * 0.25f)
#define EWALD_REAL_CUTOFF2 (2.6f*2.6f)


#endif /* COSMICTIGER_DEFS_HPP_ */
