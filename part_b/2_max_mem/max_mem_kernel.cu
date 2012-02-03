#include <cuda.h>
#include <cuda_runtime_api.h>

#define NUM_MEM_OPS_PER_KERNEL 	1024
#define N_MEMSIZE 					32
#define N_LOOPS						512

//-----------------------------------------------------------------------------
// Simple test kernel template for memory ops test
//		@param d_counters	- Simple memory location to exploit for lots of memory accesses
// 	@param n_threads	- Total number of threads per block
//-----------------------------------------------------------------------------
__global__ 
void max_flops_kernel(float* d_counters) {
    // Increment the counter
	int i = blockIdx.x * blockDim.x + threadIdx.x*N_MEMSIZE;
	// int temp = 0;
	// int temp2 = 0;
	// temp = d_counters[i];
	// d_counters[i] = d_counters[i+31];
	// temp2 = d_counters[i+1];


	for(int it = 0; it < N_LOOPS; ++it){
		d_counters[i+it] = d_counters[i+(31-it)];
	}
}
