#include <cuda.h>
#include <cuda_runtime_api.h>

#define NUM_MEM_OPS_PER_KERNEL 32

//-----------------------------------------------------------------------------
// Simple test kernel template for memory ops test
//		@param d_counters	- Simple memory location to exploit for lots of memory accesses
// 	@param n_threads	- Total number of threads per block
//-----------------------------------------------------------------------------
__global__ 
void max_flops_kernel(float* d_counters, int num_threads) {
    // Increment the counter
	int i = blockIdx.x * num_threads + threadIdx.x*NUM_MEM_OPS_PER_KERNEL;
	int temp = 0;
	int temp2 = 0;
	temp = d_counters[i];
	d_counters[i] = d_counters[i+39];
	temp2 = d_counters[i+1];
	d_counters[i+1] = temp;
	temp = d_counters[i+2];
	d_counters[i+2] = temp2;
	temp2 = d_counters[i+3];
	d_counters[i+3] = temp;
	temp = d_counters[i+4];
	d_counters[i+4] = temp2;
	temp2 = d_counters[i+5];
	d_counters[i+5] = temp;
	temp = d_counters[i+6];
	d_counters[i+6] = temp2;
	temp2 = d_counters[i+7];
	d_counters[i+7] = temp;
	temp = d_counters[i+8];
	d_counters[i+8] = temp2;
	temp2 = d_counters[i+9];
	d_counters[i+9] = temp;
	temp = d_counters[i+10];
	d_counters[i+10] = temp2;
	temp2 = d_counters[i+11];
	d_counters[i+11] = temp;
	temp = d_counters[i+12];
	d_counters[i+12] = temp2;
	temp2 = d_counters[i+13];
	d_counters[i+13] = temp;
	temp = d_counters[i+14];
	d_counters[i+14] = temp2;
	temp2 = d_counters[i+15];
	d_counters[i+15] = temp;
	temp = d_counters[i+16];
	d_counters[i+16] = temp2;
	temp2 = d_counters[i+17];
	d_counters[i+17] = temp;
	temp = d_counters[i+18];
	d_counters[i+18] = temp2;
	temp2 = d_counters[i+19];
	d_counters[i+19] = temp;
	temp = d_counters[i+20];
	d_counters[i+20] = temp2;
	temp2 = d_counters[i+21];
	d_counters[i+21] = temp;
	temp = d_counters[i+22];
	d_counters[i+22] = temp2;
	temp2 = d_counters[i+23];
	d_counters[i+23] = temp;
	temp = d_counters[i+24];
	d_counters[i+24] = temp2;
	temp2 = d_counters[i+25];
	d_counters[i+25] = temp;
	temp = d_counters[i+26];
	d_counters[i+26] = temp2;
	temp2 = d_counters[i+27];
	d_counters[i+27] = temp;
	temp = d_counters[i+28];
	d_counters[i+28] = temp2;
	temp2 = d_counters[i+29];
	d_counters[i+29] = temp;
	temp = d_counters[i+30];
	d_counters[i+30] = temp2;
	temp2 = d_counters[i+31];
	d_counters[i+31] = temp;
}
