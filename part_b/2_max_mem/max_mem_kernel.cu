#include <cuda.h>
#include <cuda_runtime_api.h>

#define NUM_MEM_OPS_PER_KERNEL 40

//-----------------------------------------------------------------------------
// Simple test kernel template for memory ops test
//		@param d_counters	- Simple memory location to exploit for lots of memory accesses
// 	@param n_threads	- Total number of threads per block
//-----------------------------------------------------------------------------
__global__ 
void max_flops_kernel(float* d_counters, int num_threads) {
    // Increment the counter
	int i = blockIdx.x * num_threads + threadIdx.x*20;
	int temp = 1;
	d_counters[i] = d_counters[i] + temp;
	d_counters[i+1] += temp;
	d_counters[i+2] += temp;
	d_counters[i+3] += temp;
	d_counters[i+4] += temp;
	d_counters[i+5] += temp;
	d_counters[i+6] += temp;
	d_counters[i+7] += temp;
	d_counters[i+8] += temp;
	d_counters[i+9] += temp;
	d_counters[i+10] += temp;
	d_counters[i+11] += temp;
	d_counters[i+12] += temp;
	d_counters[i+13] += temp;
	d_counters[i+14] += temp;
	d_counters[i+15] += temp;
	d_counters[i+16] += temp;
	d_counters[i+17] += temp;
	d_counters[i+18] += temp;
	d_counters[i+19] += temp;
	d_counters[i+20] += temp;
	d_counters[i+21] += temp;
	d_counters[i+22] += temp;
	d_counters[i+23] += temp;
	d_counters[i+24] += temp;
	d_counters[i+25] += temp;
	d_counters[i+26] += temp;
	d_counters[i+27] += temp;
	d_counters[i+28] += temp;
	d_counters[i+29] += temp;
	d_counters[i+30] += temp;
	d_counters[i+31] += temp;
	d_counters[i+32] += temp;
	d_counters[i+33] += temp;
	d_counters[i+34] += temp;
	d_counters[i+35] += temp;
	d_counters[i+36] += temp;
	d_counters[i+37] += temp;
	d_counters[i+38] += temp;
	d_counters[i+39] += temp;
}
