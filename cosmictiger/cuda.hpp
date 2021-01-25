/*
 * cuda.hpp
 *
 *  Created on: Jan 24, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_CUDA_HPP_
#define COSMICTIGER_CUDA_HPP_

#define CUDA_CHECK( a ) if( a != cudaSuccess ) printf( "CUDA error on line %i of %s : %s\n", __LINE__, __FILE__, cudaGetErrorString(a))


#endif /* COSMICTIGER_CUDA_HPP_ */
