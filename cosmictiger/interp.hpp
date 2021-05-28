/*
 * interp.hpp
 *
 *  Created on: Jan 11, 2021
 *      Author: dmarce1
 */

#ifndef GPUTIGER_INTERP_HPP_
#define GPUTIGER_INTERP_HPP_

#include <cosmictiger/vector.hpp>
#include <cuda.h>

template<class T>
struct interp_functor {
	vector<T> values;
	T amin;
	T amax;
	T minloga;
	T maxloga;
	int N;
	T dloga;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & values;
		arc & amin;
		arc & amax;
		arc & minloga;
		arc & maxloga;
		arc & N;
		arc & dloga;
	}
#ifndef __CUDA_ARCH__
	std::function<void()> to_device() {
		cudaStream_t stream;
		CUDA_CHECK(cudaStreamCreate(&stream));
		auto f = values.to_device(stream);
		CUDA_CHECK(cudaStreamSynchronize(stream));
		CUDA_CHECK(cudaStreamDestroy(stream));
		return f;
	}
#endif
	CUDA_EXPORT
	T operator()(T a) const {
		T loga = logf(a);
		if (loga < minloga || loga > maxloga) {
			PRINT("Error in interpolation_function out of range %e %e %e\n", a, amin, amax);
		}
		int i1 = int((loga - minloga) / (dloga));
		if (i1 > N - 3) {
			i1 = N - 3;
		} else if (i1 < 1) {
			i1 = 1;
		}
		int i0 = i1 - 1;
		int i2 = i1 + 1;
		int i3 = i2 + 1;
		const T c0 = values[i1];
		const T c1 = -values[i0] / (T) 3.0 - (T) 0.5 * values[i1] + values[i2] - values[i3] / (T) 6.0;
		const T c2 = (T) 0.5 * values[i0] - values[i1] + (T) 0.5 * values[i2];
		const T c3 = -values[i0] / (T) 6.0 + (T) 0.5 * values[i1] - (T) 0.5 * values[i2] + values[i3] / (T) 6.0;
		T x = (loga - i1 * dloga - minloga) / dloga;
		return c0 + c1 * x + c2 * x * x + c3 * x * x * x;
	}
};

template<class T>
CUDA_EXPORT
inline void build_interpolation_function(interp_functor<T>* f, const vector<T>& values, T amin, T amax) {
	T minloga = log(amin);
	T maxloga = log(amax);
	int N = values.size();
	T dloga = (maxloga - minloga) / (N-1);
	interp_functor<T> functor;
	functor.values = std::move(values);
	functor.maxloga = maxloga;
	functor.minloga = minloga;
	functor.dloga = dloga;
	functor.amin = amin;
	functor.amax = amax;
	functor.N = N;
	*f = functor;
}

#ifdef __CUDA_ARCH__
template<class T>
__device__
inline void build_interpolation_function(interp_functor<T>* f, T* values, T amin, T amax, int N) {
	const int& tid = threadIdx.x;
	const int& bsz = blockDim.x;
	T minloga = log(amin);
	T maxloga = log(amax);
	T dloga = (maxloga - minloga) / (N-1);
	interp_functor<T>& functor = *f;
	functor.values.resize(N);
	__syncthreads();
	for (int i = tid; i < N; i += bsz) {
		functor.values[i] = values[i];
	}
	__syncthreads();
	if( tid == 0 ) {
		functor.maxloga = maxloga;
		functor.minloga = minloga;
		functor.dloga = dloga;
		functor.amin = amin;
		functor.amax = amax;
		functor.N = N;
	}
	__syncthreads();

}
#endif

#endif /* GPUTIGER_INTERP_HPP_ */
