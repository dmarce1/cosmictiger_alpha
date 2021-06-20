
#include <hpxfft/fourier.hpp>

namespace hpxfft {

void cuda_set_device() {
	int count;
	CUDA_CHECK(cudaGetDeviceCount(&count));
	const int device_num = hpxfft::hpx_rank() % count;
	CUDA_CHECK(cudaSetDevice(device_num));
}


}
