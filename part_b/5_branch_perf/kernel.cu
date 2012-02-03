#include <cuda.h>
#include <cuda_runtime_api.h>


#define NUM_FLOPS_PER_BLOCK = FIX ME!;
//-----------------------------------------------------------------------------
// Simple test kernel template for flops test
//		@param d_counters	- Counters to hold how many FLOPs a kernel does.
// 	@param n_threads	- Total number of threads per block
//-----------------------------------------------------------------------------
__global__ 
void branch_perf_kernel(float* d_counters, int num_branches) {
    
    switch(threadIDx.x%num_branches){
    	case 0: temp = FLOPS(1);
    	case 1: temp = FLOPS(2);
    	case 2: temp = FLOPS(3);
    	case 3: temp = FLOPS(4);
    	case 4: temp = FLOPS(5);
    	case 5: temp = FLOPS(6);
    	case 6: temp = FLOPS(7);
    	case 7: temp = FLOPS(8);
    	case 8: temp = FLOPS(9);
    	case 9: temp = FLOPS(10);
    	case 10: temp = FLOPS(11);
    	case 11: temp = FLOPS(12);
    	case 12: temp = FLOPS(13);
    	case 13: temp = FLOPS(14);
    	case 14: temp = FLOPS(15);
    	case 15: temp = FLOPS(16);
    	case 16: temp = FLOPS(17);
    	case 17: temp = FLOPS(18);
    	case 18: temp = FLOPS(19);
    	case 19: temp = FLOPS(20);
    	case 20: temp = FLOPS(21);
    	case 21: temp = FLOPS(22);
    	case 22: temp = FLOPS(23);
    	case 23: temp = FLOPS(24);
    	case 24: temp = FLOPS(25);
    	case 25: temp = FLOPS(26);
    	case 26: temp = FLOPS(27);
    	case 27: temp = FLOPS(28);
    	case 28: temp = FLOPS(29);
    	case 29: temp = FLOPS(30);
    	case 30: temp = FLOPS(31);
    	case 31: temp = FLOPS(32);
    	default: temp = FLOPS(1);
    }
    
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
