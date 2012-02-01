#include <cuda.h>
#include <cuda_runtime_api.h>

//-----------------------------------------------------------------------------
// Simple test kernel template for flops test
//		@param d_counters	- Counters to hold how many FLOPs a kernel does.
// 	@param n_threads	- Total number of threads per block
//-----------------------------------------------------------------------------
__global__ 
void max_flops_kernel(float* d_counters, float n_threads) {
    // Increment the counter
	float i = blockIdx.x * n_threads + threadIdx.x;
	float temp = d_counters[(int) i] + 1.0f;
	temp = temp + 1.0f;
	temp = temp + 1.0f;
	
	temp = temp + 1.0f;
	temp = temp + 1.0f;
	temp = temp + 1.0f;
	temp = temp + 1.0f;
	temp = temp + 1.0f;
	
	temp = temp + 1.0f;
	temp = temp + 1.0f;
	temp = temp + 1.0f;
	temp = temp + 1.0f;
	temp = temp + 1.0f;
	
	temp = temp + 1.0f;
	temp = temp + 1.0f;
	temp = temp + 1.0f;
	temp = temp + 1.0f;
	temp = temp + 1.0f;
	
	temp = temp + 1.0f;
	temp = temp + 1.0f;
	temp = temp + 1.0f;
	temp = temp + 1.0f;
	d_counters[(int)i] = temp + 1.0f;
}
