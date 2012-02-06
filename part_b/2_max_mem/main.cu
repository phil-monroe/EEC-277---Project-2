#ifndef _MAX_MEM_
#define _MAX_MEM_

// Define defaults ------------------------------------------------------------
#define NUM_BLOCKS 				4096		// Keep the MPs busy for a while
#define NUM_THREADS_PER_BLOCK 384		//	Keep the cores occupied
#define BYTES_PER_FLOAT			4			// Duh...
#define OFFSET						0			// Test Coalescing
#define NUM_LOOPS					1
#define ARRAY_SIZE				(NUM_BLOCKS*NUM_THREADS_PER_BLOCK+OFFSET) 	// Combine

// includes -------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "kernel.cu"	// Kernel to Maximize FLOPS
#include "common/helpers.c"


// Forward Declarations --------------------------------------------------------
void runTest(appConfig ap);


// Main -----------------------------------------------------------------------
int main( int argc, char** argv) {
	appConfig ap = initialize(argc, argv);
	displayHeader("Memory Bandwidth Test", ap);
	
	runTest(ap);

	exit(0);
}


// runTest --------------------------------------------------------------------
//		Runs a simple test to maximize the amount of memory bandwidth available on the GPU.
// ----------------------------------------------------------------------------
void runTest(appConfig ap) {
	// Initialize arrays on host and device
	float *h_in, *h_out, *d_in, *d_out;
	initArray(&h_in,	&d_in,  ap.arraySize);
	initArray(&h_out, &d_out, ap.arraySize);
	
	// Create and Start Timer
	cudaEvent_t start, stop;
	startTest(start, stop);

	// Run the test
	for(size_t i = 0; i < ap.nLoops; ++i){
		max_flops_kernel<<< ap.nBlocks, ap.nThreadsPerBlock >>>(d_in, d_out, OFFSET);
	}

	// Get the time elapsed
	float time_s = finishTest(start, stop);
	
	// Calculate Performance
	double total_bytes = ((double)ap.nThreads * BYTES_PER_FLOAT * N_MEM_OPS_PER_KERNEL) * ap.nLoops / (1024.0*1024.0*1024.0) ;
	printf("Total GBytes Transferred : %.3f\n", total_bytes);
	double gbps = total_bytes / time_s;
	printf("GBps                     : %.3f\n", gbps);

	// Cleanup
	free(h_in);
	free(h_out);
	cudaFree(d_in);
	cudaFree(d_out);
}
#endif /* _MAX_MEM_ */
