#include <cuda.h>
#include <cuda_runtime_api.h>

#define N_LOOPS				100
#define N_FLOPS_PER_BLOCK	32
#define N_FLOPS_PER_LOOP	N_FLOPS_PER_BLOCK * 32
#define N_FLOPS_PER_KERNEL	N_FLOPS_PER_LOOP * N_LOOPS + 32

#define FLOPS_BLOCK \
			reg0  = reg1  * reg2  + reg3;  \
			reg1  = reg2  * reg3  + reg4;  \
			reg2  = reg3  * reg4  + reg5;  \
			reg3  = reg4  * reg5  + reg6;  \
			reg4  = reg5  * reg6  + reg7;  \
			reg5  = reg6  * reg7  + reg8;  \
			reg6  = reg7  * reg8  + reg9;  \
			reg7  = reg8  * reg9  + reg10; \
			reg8  = reg9  * reg10 + reg11; \
			reg9  = reg10 * reg11 + reg12; \
			reg10 = reg11 * reg12 + reg13; \
			reg11 = reg12 * reg13 + reg14; \
			reg12 = reg13 * reg14 + reg15; \
			reg13 = reg14 * reg15 + reg0;  \
			reg14 = reg15 * reg0  + reg1;  \
			reg15 = reg0  * reg1  + reg2;

//-----------------------------------------------------------------------------
// Simple test kernel template for flops test
//		@param d_counters	- Counters to hold how many FLOPs a kernel does.
// 	@param n_threads	- Total number of threads per block
//-----------------------------------------------------------------------------
__global__ 
void max_flops_kernel(float* d_counters) {
    // Increment the counter
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Declare a bunch or registers and init to 0.0f
	float reg0, reg1, reg2,  reg3,  reg4,  reg5,  reg6,  reg7, \
			reg8, reg9, reg10, reg11, reg12, reg13, reg14, reg15;
	
	// 1 FLOP per assignment = 16 FLOPs total
	reg0  = reg1  = reg2  = reg3  = 0.125f * threadIdx.x;
	reg4  = reg5  = reg6  = reg7  = 0.125f * threadIdx.y;
	reg8  = reg9  = reg10 = reg11 = 0.5f 	* threadIdx.x;
	reg12 = reg13 = reg14 = reg15 = 0.5f 	* threadIdx.y;
	
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

	// 16 More flops.
	d_counters[i] = reg0 + reg1 + reg2  + reg3  + reg4  + reg5  + reg6  + reg7  + reg8 \
								+ reg9 + reg10 + reg11 + reg12 + reg13 + reg14 + reg15 + 8.0f;
}
