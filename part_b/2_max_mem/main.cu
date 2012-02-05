#ifndef _MAX_MEM_
#define _MAX_MEM_

// includes -------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "kernel.cu"	// Kernel to Maximize FLOPS
#include "common/helpers.c"

// Defines --------------------------------------------------------------------
#define NUM_BLOCKS 				4096		// Keep the MPs busy for a while
#define NUM_THREADS_PER_BLOCK 384		//	Keep the cores occupied
#define BYTES_PER_FLOAT			4			// Duh...
#define OFFSET						0			// Test Coalescing
#define ARRAY_SIZE				(NUM_BLOCKS*NUM_THREADS_PER_BLOCK+OFFSET) 	// Combine


// Forward Declarations --------------------------------------------------------
void runTest( int argc, char** argv);

// Main -----------------------------------------------------------------------
int main( int argc, char** argv) {
	appConfig.nBlocks 			= NUM_BLOCKS;
	appConfig.nThreadsPerBlock = NUM_THREADS_PER_BLOCK;
	appConfig.arraySize 			= ARRAY_SIZE;
	appConfig.nThreads 			= appConfig.nBlocks * appConfig.nThreadsPerBlock;
	
	displayHeader("Memory Bandwidth Test");
	
	runTest(argc, argv);

	exit(0);
}

// runTest --------------------------------------------------------------------
//		Runs a simple test to maximize the amount of memory bandwidth available on the GPU.
//
void runTest( int argc, char** argv) {
	// Initialize arrays on host and device
	float *h_in, *h_out, *d_in, *d_out;
	initArray(&h_in,	&d_in,  appConfig.arraySize);
	initArray(&h_out, &d_out, appConfig.arraySize);
	
	// Create and Start Timer
	cudaEvent_t start, stop;
	startTest(start, stop);

	// Run the test
	max_flops_kernel<<< appConfig.nBlocks, appConfig.nThreadsPerBlock >>>(d_in, d_out, OFFSET);

	// Get the time elapsed
	float time_s = finishTest(start, stop);
	
	// Calculate Performance
	double total_bytes = ((double)appConfig.nThreads * BYTES_PER_FLOAT * N_MEM_OPS_PER_KERNEL) / (1024.0*1024.0*1024.0) ;
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
