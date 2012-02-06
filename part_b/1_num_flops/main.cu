#ifndef _MAX_FLOPS_
#define _MAX_FLOPS_

// Define Defaults ------------------------------------------------------------
#define NUM_BLOCKS 				128
#define NUM_THREADS_PER_BLOCK 128	//	Taken from CUDA Occupancy Calc to maximize occupancy
#define NUM_LOOPS					1
#define ARRAY_SIZE				(NUM_BLOCKS*NUM_THREADS_PER_BLOCK) 	// Combine


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
void runTest( appConfig ap );

// Main -----------------------------------------------------------------------
int main( int argc, char** argv) {
	appConfig ap = initialize(argc, argv);
	displayHeader("Maximum FLOPS Test", ap);
	
	runTest(ap);

	exit(0);
}

// runTest --------------------------------------------------------------------
//		Runs a simple test to maximize the number of FLOPS computed on the GPU.
//
void runTest( appConfig ap ) {
	// Initialize counters on host and device to 0.0f
	float *h_counters, *d_counters;
	initArray(&h_counters, &d_counters, ap.arraySize);


	// Create and Start Timer
	cudaEvent_t start, stop;
	startTest(start, stop);

	// Run the test
	for(size_t i = 0; i < ap.nLoops; ++i){
		kernel<<< ap.nBlocks, ap.nThreadsPerBlock>>>(d_counters);	
	}
	
	// Get the time elapsed
	float time_s = finishTest(start, stop);

	// Calculate GFLOPS
	unsigned long long total_flops = ap.nThreads * ap.nLoops * N_FLOPS_PER_KERNEL;
	printf("Total FLOPs: %lld\n", total_flops);
	float gflops = total_flops/(time_s*1024.0*1024.0*1024.0);
	printf("GFLOPS: %.3f\n", gflops);

	// Cleanup
	free(h_counters);
	cudaFree(d_counters);
}


#endif /* MAX_FLOPS */
