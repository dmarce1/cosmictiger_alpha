/*
 * print.hpp
 *
 *  Created on: Jun 20, 2021
 *      Author: dmarce1
 */

#ifndef PRINT_HPP_
#define PRINT_HPP_




#define PRINT(...) print(__VA_ARGS__)


#include <stdio.h>




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



#endif /* PRINT_HPP_ */
