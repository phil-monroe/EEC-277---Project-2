#ifndef _WARPS_V_PERF_
#define _WARPS_V_PERF_

// Define Defaults ------------------------------------------------------------
// Hardware Dependent - NV GeForce 9500 GT

#define NUM_THREADS_PER_BLOCK 0	//	Gets iterated over
#define ARRAY_SIZE				0 	// Changed per iteration

#define NUM_BLOCKS				4
#define N_THREADS_PER_WARP		32
#define NUM_LOOPS					16

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
	printf("Testing Number of Warps vs. Performance\n");
	printf("Written by Phil Monroe and Kramer Straube\n");
	printf("\n");

	FILE *file; 
	file = fopen("out.csv","a+");
	
	for(int iter = 0; iter < NUM_LOOPS; ++iter){
		printf("Iteration %d\n", iter);
		float perf = runTest((iter+1) * N_THREADS_PER_WARP); 
		fprintf(file, "%d, %d, %f\n", iter, iter+1, perf);
	}
	
	fclose(file);
	exit(0);
}

// runTest --------------------------------------------------------------------
//		Runs a simple test to determine the Blocks per Second computed for a given
//		number of blocks
//
float runTest( int num_threads) {
	printf("Testing %4d Threads\n", num_threads);
	

	// Initialize counters on host and device to 0.0f
	float *h_counters, *d_counters;
	initArray(&h_counters, &d_counters, num_threads);

	// Create and Start Timer
	cudaEvent_t start, stop;
	startTest(start, stop);

	// Run the test
	warps_v_perf_kernel<<< NUM_BLOCKS, num_threads>>>(d_counters);
	
	// Get the time elapsed
	float time_s = finishTest(start, stop);

	// Calculate Performance
	unsigned long long total_flops = (long long)N_FLOPS_PER_KERNEL * (long long) NUM_BLOCKS * num_threads;
	printf("Total FLOPs: %lld\n", total_flops);
	float gflops = total_flops/(time_s*1000000000.0f);
	printf("GFLOPS: %.3f\n", gflops);
		
	float perf = num_threads/(time_s* 1000.0f);
	printf("Threads per Sec.: %.3f KThreads/s\n", perf);
	printf("\n");
	
	// Cleanup
	free(h_counters);
	cudaFree(d_counters);
	
	return gflops;
}

#endif /* _WARPS_V_PERF_ */
