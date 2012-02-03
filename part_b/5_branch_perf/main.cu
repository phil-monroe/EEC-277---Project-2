#ifndef _MAX_FLOPS_
#define _MAX_FLOPS_

// includes -------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "kernel.cu"	// Kernel to Maximize FLOPS

// Defines --------------------------------------------------------------------
#define NUM_BLOCKS 				128
#define NUM_THREADS_PER_BLOCK 128	//	Taken from CUDA Occupancy Calc to maximize occupancy
#define NUM_ITERATIONS 			32

// Forward Declarations --------------------------------------------------------
void runTest( int num_branches);
void init_counters(float** h_counters, float** d_counters, unsigned int num_counters);

// Main -----------------------------------------------------------------------
int main( int argc, char** argv) {
	printf("Testing Number of Branched vs. Performance");
	printf("Number of Threads/Blocks: %4d\n", NUM_THREADS_PER_BLOCK);
	printf("\n");
	
	// FILE *file; 
	// file = fopen("out.csv","a+");
	
	for(int iter = 0; iter < NUM_ITERATIONS; ++iter){
		printf("Iteration %d\n", iter);
		runTest(iter+1); 
		// fprintf(file, "%d, %d, %f\n", iter, iter+1, perf);
	}
	
	// fclose(file);
	exit(0);
}

// runTest --------------------------------------------------------------------
//		Runs a simple test to maximize the number of FLOPS computed on the GPU.
//
void runTest( int num_branches) {
	printf("Number of branches: %d \n", num_branches);
	// Hardware Dependent - NV GeForce 9500 GT
	unsigned int threads = NUM_BLOCKS * NUM_THREADS_PER_BLOCK;	

	// Initialize counters on host and device to 0.0f
	printf("Init counters\n");
	float *h_counters, *d_counters;
	init_counters(&h_counters, &d_counters, threads);


	// Create and Start Timer
	printf("Starting Test\n");
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );

	// Run the test
	branch_perf_kernel<<< NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_counters, num_branches);	
	
	// Record end time
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	float time_s = time/1000.0f;
	printf("Finished Test in %f s\n", time_s);

	// Check for errors
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString( error ));
		
	// Check array
	cudaMemcpy(h_counters, d_counters, threads * sizeof(float), cudaMemcpyDeviceToHost);
	// 
	// for(int i = 0; i < threads; ++i){
	// 	printf("Thread %d: %f\n", i, h_counters[i]);
	// }

	// Calculate GFLOPS
	unsigned long long total_flops = N_FLOPS_PER_THREAD * NUM_BLOCKS * NUM_THREADS_PER_BLOCK;
	printf("Total FLOPs: %lld\n", total_flops);
	float gflops = total_flops/(time_s*1000000000.0f);
	printf("GFLOPS: %.3f\n", gflops);
	
	printf("\n");
	


	// Cleanup
	free(h_counters);
	cudaFree(d_counters);
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