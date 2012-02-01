#include <cuda.h>
#include <cuda_runtime_api.h>

//-----------------------------------------------------------------------------
// Simple test kernel template for flops test
//		@param g_idata  input data in global memory
// 	@param g_odata  output data in global memory
//-----------------------------------------------------------------------------
__global__ 
void max_flops_kernel(float* d_counters, int n_threads) {
    // Increment the counter
	int i = blockIdx.x * n_threads + threadIdx.x;
    d_counters[i] = d_counters[i] + 1.0f;
}
