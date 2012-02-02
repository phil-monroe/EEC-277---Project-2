#ifndef _BLOCKS_V_PERF_
#define _BLOCKS_V_PERF_

// includes -------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "kernel.cu"	// Kernel to Maximize FLOPS

// Defines --------------------------------------------------------------------
// Hardware Dependent - NV GeForce 9500 GT

#define NUM_THREADS_PER_BLOCK 384	//	Taken from CUDA Occupancy Calc to maximize occupancy
#define NUM_ITERATIONS			16

// Forward Declarations --------------------------------------------------------
void init_counters(float** h_counters, float** d_counters, unsigned int num_counters);
float runTest(int num_blocks);

// Main -----------------------------------------------------------------------
int main( int argc, char** argv) {
	printf("Testing Number of Blocks vs. Performance");
	printf("Number of Threads/Blocks: %4d\n", NUM_THREADS_PER_BLOCK);
	printf("\n");
	
	FILE *file; 
	file = fopen("out.csv","a+");
	
	for(int iter = 0; iter < NUM_ITERATIONS; ++iter){
		printf("Iteration %d\n", iter);
		int num_blocks = (int) pow(2.0f, (float) iter); 	// number of blocks to run is 2^iter
		float perf = runTest(num_blocks); 
		fprintf(file, "%d, %d, %f\n", iter, num_blocks, perf);
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
	init_counters(&h_counters, &d_counters, num_threads);


	// Create and Start Timer
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );

	// Run the test
	block_perf_kernel<<< num_blocks, NUM_THREADS_PER_BLOCK>>>(d_counters, NUM_THREADS_PER_BLOCK);
	cudaThreadSynchronize(); // Make sure all GPU computations are done
	
	
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
	//cudaMemcpy(h_counters, d_counters, threads * sizeof(float), cudaMemcpyDeviceToHost);
	
	// for(int i = 0; i < threads; ++i){
	// 	printf("Thread %d: %f\n", i, h_counters[i]);
	// }

	// Calculate Performance
	float perf = num_blocks/(time_s* 1000.0f);
	printf("Total Perf: %.3f KBlocks/s\n", perf);
	printf("\n");
	
	// Cleanup
	free(h_counters);
	cudaFree(d_counters);
	
	return perf;
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


#endif /* _BLOCKS_V_PERF_ */
