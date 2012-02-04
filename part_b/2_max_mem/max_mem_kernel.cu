#include <cuda.h>
#include <cuda_runtime_api.h>

#define N_MEMSIZE_PER_THREAD		1024
#define N_MEM_OPS_PER_LOOP			4
#define N_LOOPS						(N_MEMSIZE_PER_THREAD / 2)
#define NUM_MEM_OPS_PER_KERNEL 	(N_MEM_OPS_PER_LOOP * N_LOOPS)

//-----------------------------------------------------------------------------
// Simple test kernel template for memory ops test
//		@param d_counters	- Simple memory location to exploit for lots of memory accesses
// 	@param n_threads	- Total number of threads per block
//-----------------------------------------------------------------------------
__global__ 
void max_flops_kernel(float* d_counters) {
    // Increment the counter
	int i = blockIdx.x * blockDim.x + threadIdx.x*N_MEMSIZE_PER_THREAD;
	float temp;
	for(int it = 0; it < N_MEMSIZE_PER_THREAD; ++it){
		temp = d_counters[i+it];
		d_counters[i+it] = d_counters[i+(N_MEMSIZE_PER_THREAD-it)];
		d_counters[i+(N_MEMSIZE_PER_THREAD-it)] = temp;
	}
}
