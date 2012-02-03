def _BRANCHES_V_PERF_
#define _BRANCHES_V_PERF_

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
#define NUM_BRANCHES_PER_THREAD 1

// Forward Declarations --------------------------------------------------------
void init_counters(float** h_counters, float** d_counters, unsigned int num_counters);
float runTest(int num_blocks);

// Main -----------------------------------------------------------------------
int main( int argc, char** argv) {
	printf("Testing Number of Branches vs. Performance");
	printf("Number of Threads/Blocks: %4d\n", NUM_THREADS_PER_BLOCK);
	printf("\n");
	
	FILE *file; 
	file = fopen("out.csv","a+");
	
	for(int iter = 0; iter < NUM_ITERATIONS; ++iter){
		printf("Iteration %d\n", iter);
		int num_branches = (int) pow(2.0f, (float) iter); 	// number of branches to run is 2^iter
		float perf = runTest(num_branches); 
		fprintf(file, "%d, %d, %f\n", iter, num_branches, perf);
	}
	
	fclose(file);
	exit(0);
}

// runTest --------------------------------------------------------------------
//		Runs a simple test to determine the FLOPS computed for a given
//		number of blocks
//
float runTest( int num_branches) {
	
	printf("Testing %4d Branches\n", num_branches);
	int num_threads = NUM_BLOCKS * NUM_THREADS_PER_BLOCK;
	int branch_gran = 32/num_branches;
	

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
	branch_perf_kernel<<< NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_counters, num_branches);
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
	float perf = NUM_FLOPS_PER_BLOCK*NUM_BLOCKS/(time_s* 1000.0f);
	printf("Total Perf: %.3f FLOPS\n", perf);
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
