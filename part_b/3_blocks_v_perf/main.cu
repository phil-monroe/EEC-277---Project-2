#ifndef _BLOCKS_V_PERF_
#define _BLOCKS_V_PERF_

// Defines --------------------------------------------------------------------
// Hardware Dependent - NV GeForce 9500 GT

#define NUM_BLOCKS				0		// Gets iterated over
#define ARRAY_SIZE				0		//	Gets changed with every iteration
#define NUM_THREADS_PER_BLOCK 384	//	Taken from CUDA Occupancy Calc to maximize occupancy
#define NUM_LOOPS					100

// includes -------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "kernel.cu"	// Kernel to Maximize FLOPS
#include "common/helpers.c"



// Forward Declarations --------------------------------------------------------
float runTest(int num_blocks);

// Main -----------------------------------------------------------------------
int main( int argc, char** argv) {
	printf("Testing Number of Blocks vs. Performance\n");
	printf("Written by Phil Monroe and Kramer Straube\n\n");
	printf("Number of Threads/Blocks: %4d\n\n", NUM_THREADS_PER_BLOCK);
	
	FILE *file; 
	file = fopen("out.csv","a+");
	
	for(int iter = 0; iter < NUM_LOOPS; ++iter){
		printf("Iteration %d\n", iter);
		float perf = runTest(iter+1); 
		fprintf(file, "%d, %d, %f\n", iter, iter+1, perf);
	}
	
	fclose(file);
	exit(0);
}

// runTest --------------------------------------------------------------------
//		Runs a simple test to determine the Blocks per Second computed for a given
//		number of blocks
//
float runTest( int num_blocks) {
	
	printf("Testing %4d Blocks\n", num_blocks);
	int num_threads = num_blocks * NUM_THREADS_PER_BLOCK;

	// Initialize counters on host and device to 0.0f
	float *h_counters, *d_counters;
	initArray(&h_counters, &d_counters, num_threads);

	// Create and Start Timer
	cudaEvent_t start, stop;
	startTest(start, stop);

	// Run the test
	max_flops_kernel<<< num_blocks, NUM_THREADS_PER_BLOCK>>>(d_counters);
	
	// Get the time elapsed
	float time_s = finishTest(start, stop);

	// Calculate Performance
	unsigned long long total_flops = (long long)N_FLOPS_PER_KERNEL * (long long)num_blocks * (long long)NUM_THREADS_PER_BLOCK;
	printf("Total FLOPs: %lld\n", total_flops);
	float gflops = total_flops/(time_s*1000000000.0f);
	printf("GFLOPS: %.3f\n", gflops);
		
	float perf = num_blocks/(time_s* 1000.0f);
	printf("Blocks per Sec.: %.3f KBlocks/s\n", perf);
	printf("\n");
	
	// Cleanup
	free(h_counters);
	cudaFree(d_counters);
	
	return gflops;
}


#endif /* _BLOCKS_V_PERF_ */
