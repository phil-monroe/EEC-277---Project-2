#ifndef _MAX_FLOPS_
#define _MAX_FLOPS_

// Define Defaults ------------------------------------------------------------
#define NUM_BLOCKS 				128
#define NUM_THREADS_PER_BLOCK 128	//	Taken from CUDA Occupancy Calc to maximize occupancy
#define NUM_LOOPS		 			32
#define ARRAY_SIZE				(NUM_BLOCKS * NUM_THREADS_PER_BLOCK)

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
float runTest( int num_branches);
void init_counters(float** h_counters, float** d_counters, unsigned int num_counters);

// Main -----------------------------------------------------------------------
int main( int argc, char** argv) {
	printf("Testing Number of Branched vs. Performance\n");
	printf("Written by Phil Monroe and Kramer Straube\n");
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
//		Runs a simple test to maximize the number of FLOPS computed on the GPU.
//
float runTest( int num_branches) {
	printf("Number of branches: %d \n", num_branches);
	unsigned int threads = NUM_BLOCKS * NUM_THREADS_PER_BLOCK;	

	// Initialize counters on host and device to 0.0f
	printf("Init counters\n");
	float *h_counters, *d_counters;
	initArray(&h_counters, &d_counters, threads);


	// Create and Start Timer
	cudaEvent_t start, stop;
	startTest(start, stop);

	// Run the test
	branch_perf_kernel<<< NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_counters, num_branches);	
	
	// Get the time elapsed
	float time_s = finishTest(start, stop);

	// Calculate GFLOPS
	unsigned long long total_flops = N_FLOPS_PER_THREAD * NUM_BLOCKS * NUM_THREADS_PER_BLOCK;
	printf("Total FLOPs: %lld\n", total_flops);
	float gflops = total_flops/(time_s*1000000000.0f);
	printf("GFLOPS: %.3f\n", gflops);

	// Cleanup
	free(h_counters);
	cudaFree(d_counters);
	return gflops;
}

// init_counters --------------------------------------------------------------
//		Initializes an array of floats that will be used to count FLOPS.
//
void init_counters(float** h_counters, float** d_counters, unsigned int num_counters){
	*h_counters = (float*) malloc( num_counters * sizeof(float));   // Allocate counters on host
	cudaMalloc((void **) d_counters, num_counters*sizeof(float));   // Allocate counters on device

	// Initialize host counters to 0.0 ...
	for( unsigned int i = 0; i < num_counters; ++i)
		(*h_counters)[i] = 0.0f;
	// ... and copy to device
	cudaMemcpy(*d_counters, *h_counters, num_counters * sizeof(float), cudaMemcpyHostToDevice);
}


#endif /* MAX_FLOPS */
