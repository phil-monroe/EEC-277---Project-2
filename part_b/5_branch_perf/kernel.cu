#include <cuda.h>
#include <cuda_runtime_api.h>

#define N_FLOPS_PER_THREAD 	784
#define N_LOOPS 					1



#define FLOPS_BLOCK \
reg0 = reg1  * reg2  + reg3;  \
	reg5 = reg6 * reg6;				\
	reg1 = reg2  * reg3  + reg4;  \
	reg6 = reg7 * reg7;				\
	reg2 = reg3  * reg4  + reg5;  \
	reg7 = reg0 * reg0;				\
	reg3 = reg4  * reg5  + reg6;  \
	reg0 = reg1 * reg1;				\
	reg4 = reg5  * reg6  + reg7;  \
	reg1 = reg2 * reg2;				\
	reg5 = reg6  * reg7  + reg0;  \
	reg2 = reg3 * reg3;				\
	reg6 = reg7  * reg0  + reg1;  \
	reg3 = reg4 * reg4;				\
	reg7 = reg0  * reg1  + reg2;  \
	reg4 = reg5 * reg5;



__device__ float flops(int i){
	// Declare a bunch or registers and init to 0.0f
	float reg0, reg1, reg2,  reg3,  reg4,  reg5,  reg6,  reg7;

	// 1 FLOP per assignment = 8 FLOPs total
	reg0  = reg1  = reg2  = reg3  = 9.765625e-10f * threadIdx.x;
	reg4  = reg5  = reg6  = reg7  = 9.765625e-10f * threadIdx.y;

	for(int it = 0; it < N_LOOPS; ++it){
		FLOPS_BLOCK // 1
		FLOPS_BLOCK
		FLOPS_BLOCK
		FLOPS_BLOCK
		FLOPS_BLOCK
		FLOPS_BLOCK
		FLOPS_BLOCK
		FLOPS_BLOCK // 8
		FLOPS_BLOCK
		FLOPS_BLOCK
		FLOPS_BLOCK
		FLOPS_BLOCK
		FLOPS_BLOCK
		FLOPS_BLOCK
		FLOPS_BLOCK
		FLOPS_BLOCK // 16
		FLOPS_BLOCK
		FLOPS_BLOCK
		FLOPS_BLOCK
		FLOPS_BLOCK
		FLOPS_BLOCK
		FLOPS_BLOCK
		FLOPS_BLOCK
		FLOPS_BLOCK // 24
		FLOPS_BLOCK
		FLOPS_BLOCK
		FLOPS_BLOCK
		FLOPS_BLOCK
		FLOPS_BLOCK
		FLOPS_BLOCK
		FLOPS_BLOCK
		FLOPS_BLOCK  // 32
	}

	return reg0 + reg1 + reg2  + reg3  + reg4  + reg5  + reg6  + reg7  +  i;
}


//-----------------------------------------------------------------------------
// Simple test kernel template for flops test
//		@param d_counters	- Counters to hold how many FLOPs a kernel does.
// 	@param n_threads	- Total number of threads per block
//-----------------------------------------------------------------------------
__global__ 
void branch_perf_kernel(float* d_counters, int num_branches) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float temp; 
	switch((int)threadIdx.x%num_branches){
		case 0: temp =  flops(1); 	break;
		case 1: temp =  flops(2);  break;
		case 2: temp =  flops(3);  break;
		case 3: temp =  flops(4);  break;
		case 4: temp =  flops(5);  break;
		case 5: temp =  flops(6);  break;
		case 6: temp =  flops(7);  break;
		case 7: temp =  flops(8);  break;
		case 8: temp =  flops(9);  break;
		case 9: temp =  flops(10); break;
		case 10: temp = flops(11); break;
		case 11: temp = flops(12); break;
		case 12: temp = flops(13); break;
		case 13: temp = flops(14); break;
		case 14: temp = flops(15); break;
		case 15: temp = flops(16); break;
		case 16: temp = flops(17); break;
		case 17: temp = flops(18); break;
		case 18: temp = flops(19); break;
		case 19: temp = flops(20); break;
		case 20: temp = flops(21); break;
		case 21: temp = flops(22); break;
		case 22: temp = flops(23); break;
		case 23: temp = flops(24); break;
		case 24: temp = flops(25); break;
		case 25: temp = flops(26); break;
		case 26: temp = flops(27); break;
		case 27: temp = flops(28); break;
		case 28: temp = flops(29); break;
		case 29: temp = flops(30); break;
		case 30: temp = flops(31); break;
		case 31: temp = flops(32); break;
		default: temp = 0.0f;
	}
	d_counters[i] = temp;
}

