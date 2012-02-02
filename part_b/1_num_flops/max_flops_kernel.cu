#include <cuda.h>
#include <cuda_runtime_api.h>

#define NUM_FLOPS_PER_KERNEL 100

//-----------------------------------------------------------------------------
// Simple test kernel template for flops test
//		@param d_counters	- Counters to hold how many FLOPs a kernel does.
// 	@param n_threads	- Total number of threads per block
//-----------------------------------------------------------------------------
__global__ 
void max_flops_kernel(float* d_counters, int num_threads) {
    // Increment the counter
	int i = blockIdx.x * num_threads + threadIdx.x;
	float temp = 1.0f;
	
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;
	temp = temp + .05f * temp;

	
	d_counters[i] = temp;
}
