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
#endif
//#define TEST_STACK

#define TREE_MAX_DEPTH 52
#define TREE_RADIX_CUSHION -2
#define TREE_RADIX_MIN 3
#define TREE_RADIX_MAX 52
#define TREE_SORT_MULTITHREAD
#define TREE_MIN_PARTS2THREAD (64*1024)


#define NCHILD 2
//#define PARALLEL_RADIX

#define ALLOCATION_PAGE_SIZE (2*1024*1024LL)
#define MIN_CUDA_SORT 65536LL
#define OVERSUBSCRIPTION 8

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
