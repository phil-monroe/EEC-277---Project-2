#include <cuda.h>
#include <cuda_runtime_api.h>

//-----------------------------------------------------------------------------
// Simple test kernel template for flops test
//		@param d_counters	- Counters to hold how many FLOPs a kernel does.
// 	@param n_threads	- Total number of threads per block
//-----------------------------------------------------------------------------
__global__ 
void block_perf_kernel(float* d_counters, int num_threads) {
    // Increment the counter
	int i = blockIdx.x * num_threads + threadIdx.x;
	float temp = 1.0f;
	
	for(size_t i = 0; i < 1000; ++i){
		temp = temp + .05f * temp;
	}
	for(size_t i = 0; i < 1000; ++i){
		temp = temp - .2f * temp;
	}

	d_counters[i] = temp;
}
