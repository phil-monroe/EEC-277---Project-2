#ifndef _MAX_FLOPS_
#define _MAX_FLOPS_

// includes -------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "max_flops_kernel.cu"	// Kernel to Maximize FLOPS



// Forward Declarations --------------------------------------------------------
void runTest( int argc, char** argv);
void init_counters(float** h_counters, float** d_counters, unsigned int num_counters);

// Main -----------------------------------------------------------------------
int main( int argc, char** argv) {
	runTest( argc, argv);

	exit(0);
}

// runTest --------------------------------------------------------------------
//		Runs a simple test to maximize the number of FLOPS computed on the GPU.
//
void runTest( int argc, char** argv) {

	// Hardware Dependent - NV GeForce 9500 GT
	unsigned int dev, blocks, threads_per_block, threads;
	dev = 0;
	blocks = 512;
	threads_per_block = 512;
	threads = blocks * threads_per_block;

	// Initialize counters on host and device to 0.0f
	fprintf(stderr, "initializing counters\n");
	float *h_counters, *d_counters;
	init_counters(&h_counters, &d_counters, threads);

	// Create and Start Timer
	fprintf(stderr, "Starting Test\n");
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );
	for(size_t i = 0; i < 10; ++i){
		// Run FLOPS
		max_flops_kernel<<< blocks, threads_per_block>>>(d_counters, threads_per_block);
	}
	
	// max_flops_kernel<<< blocks, threads_per_block>>>(d_counters, threads_per_block);

	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	fprintf(stderr, "Finished Test in %f s\n", time/1000.0f);

	// Check for errors
	printf("Error: %s\n", cudaGetErrorString( cudaGetLastError()));

	// Copy Memory back
	cudaMemcpy(h_counters, d_counters, threads * sizeof(float), cudaMemcpyDeviceToHost);


	// Count up counters
	float count = 0.0f;
	for(int i = 0; i < threads; ++i){
		printf("Thread %5d OPs: %f\n", i, h_counters[i]);
		count = count + h_counters[i];
	}
	printf("Total OPs: %d\n", (int) count);
	float gflops = (int) count/(time/1000)/1000/1000/1000;


	printf("GFLOPS: %.2f\n", gflops);


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
