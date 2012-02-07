#ifndef _MAX_FLOPS_
#define _MAX_FLOPS_

// Define Defaults ------------------------------------------------------------
#define NUM_BLOCKS 				128
#define NUM_THREADS_PER_BLOCK 128		//	Taken from CUDA Occupancy Calc to maximize occupancy
#define NUM_LOOPS					1
#define ARRAY_SIZE				16384 	// Combine


// includes -------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "single_kernel.cu"	// One thread does one piece of work
#include "multi_kernel.cu"		// One thread does multiple pieces of work

#include "common/helpers.c"


// Forward Declarations --------------------------------------------------------
void runTest( appConfig ap );

// Main -----------------------------------------------------------------------
int main( int argc, char** argv) {
	appConfig ap = initialize(argc, argv);
	displayHeader("Thread Work Comparison Test", ap);
	
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


	// Create and Start Timer for single kernel
	cudaEvent_t start, stop;
	startTest(start, stop, "Starting single job per thread test.");

	// Run the test
	for(size_t i = 0; i < ap.nLoops; ++i){
		single_kernel<<< ap.nBlocks, ap.nThreadsPerBlock>>>(d_counters);	
	}
	
	// Get the time elapsed
	float time_s = finishTest(start, stop);

	// Calculate GFLOPS
	unsigned long long total_flops = ap.nThreads * ap.nLoops * N_FLOPS_PER_KERNEL;
	printf("Total FLOPs: %lld\n", total_flops);
	float gflops = total_flops/(time_s*1000000000.0f);
	printf("GFLOPS:      %.3f\n", gflops);
	printf("\n");
	
	printf("Number of Blocks:           4\n");
	printf("Number of Threads/Block:  256\n");
	printf("Number of Threads:       1024\n");
	printf("\n");
	
	// Create and Start Timer for multi kernel
	startTest(start, stop, "Starting multiple jobs per thread test.");

	// Run the test
	for(size_t i = 0; i < ap.nLoops; ++i){
		multi_kernel<<< 4, 256>>>(d_counters);	
	}
	
	// Get the time elapsed
	time_s = finishTest(start, stop);

	// Calculate GFLOPS
	total_flops = 256.0f * 4.0f * N_FLOPS_PER_KERNEL * 16.0f;
	printf("Total FLOPs: %lld\n", total_flops);
	gflops = total_flops/(time_s*1000000000.0f);
	printf("GFLOPS:      %.3f\n", gflops);

	// Cleanup
	free(h_counters);
	cudaFree(d_counters);
}


#endif /* MAX_FLOPS */
