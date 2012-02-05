#include <cuda.h>
#include <cuda_runtime_api.h>

#define N_MEM_OPS_PER_KERNEL 	2

//-----------------------------------------------------------------------------
// Simple test kernel template for memory ops test
//		@param d_counters	- Simple memory location to exploit for lots of memory accesses
// 	@param n_threads	- Total number of threads per block
//-----------------------------------------------------------------------------
__global__ 
void max_flops_kernel(float* d_in, float* d_out, int offset) {
    // Increment the counter
	int it = blockIdx.x * blockDim.x + threadIdx.x + offset;  
	d_out[it] = d_in[it];
}
