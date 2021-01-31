/*
 * defs.hpp
 *
 *  Created on: Jan 23, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_DEFS_HPP_
#define COSMICTIGER_DEFS_HPP_

#define NDIM 3

//#define TEST_STACK
//#define TEST_TREE
//#define TEST_RADIX

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
#define OVERSUBSCRIPTION 2

template<class T>
inline T sqr(T a) {
   return a * a;
}

#endif /* COSMICTIGER_DEFS_HPP_ */
