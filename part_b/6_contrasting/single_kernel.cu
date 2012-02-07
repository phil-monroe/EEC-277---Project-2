#include <cuda.h>
#include <cuda_runtime_api.h>

#define N_LOOPS				100
#define N_FLOPS_PER_BLOCK	24
#define N_FLOPS_PER_LOOP	76800
#define N_FLOPS_PER_KERNEL	76816

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
			reg4 = reg5 * reg5;				\

//-----------------------------------------------------------------------------
// Simple test kernel template for flops test
//		@param d_counters	- Counters to hold how many FLOPs a kernel does.
// 	@param n_threads	- Total number of threads per block
//-----------------------------------------------------------------------------
__global__ 
void single_kernel(float* d_counters) {
    // Increment the counter
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Declare a bunch or registers and init to 0.0f
	float reg0, reg1, reg2,  reg3,  reg4,  reg5,  reg6,  reg7;
	
	// 1 FLOP per assignment = 8 FLOPs total
	reg0  = reg1  = reg2  = reg3  = 9.765625e-10f * threadIdx.x;
	reg4  = reg5  = reg6  = reg7  = 9.765625e-10f * threadIdx.y;
	
	for(int i = 0; i < N_LOOPS; ++i){
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

	// 8 More flops.
	d_counters[i] = reg0 + reg1 + reg2  + reg3  + reg4  + reg5  + reg6  + reg7  + 8.0f;
}
