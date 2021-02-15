/*
 * defs.hpp
 *
 *  Created on: Jan 23, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_DEFS_HPP_
#define COSMICTIGER_DEFS_HPP_

#define NDIM 3

//#define TEST_CHECKLIST_TIME

#ifndef NDEBUG
#define TEST_TREE
#define TEST_RADIX
#define TEST_BOUNDS
//#define TEST_FORCE
#endif
//#define TEST_STACK

#define TREE_MAX_DEPTH 54
#define TREE_RADIX_CUSHION -2
#define TREE_RADIX_MIN 3
#define TREE_RADIX_MAX 54
#define TREE_SORT_MULTITHREAD
#define TREE_MIN_PARTS2THREAD (64*1024)
#define WORKSPACE_SIZE 1024
#define KICK_GRID_SIZE (1024)
#define KICK_BLOCK_SIZE 32
#define KICK_PP_MAX size_t(256)
#define GPU_QUEUE_SIZE (1024*1024)
#define MAX_BUCKET_SIZE 64



#ifdef ACCUMULATE_DOUBLE_PRECISION
using accum_real = double;
#else
using accum_real = float;
#endif


#ifdef EWALD_DOUBLE_PRECISION
using ewald_real = double;
#define EWALD_NREAL 17
#define EWALD_REAL_CUTOFF 3.6
#define EWALD_NFOUR 10
#else
#define EWALD_NREAL 13
#define EWALD_REAL_CUTOFF 2.6
#define EWALD_NFOUR 8
using ewald_real = float;
#endif

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



#endif /* COSMICTIGER_DEFS_HPP_ */
