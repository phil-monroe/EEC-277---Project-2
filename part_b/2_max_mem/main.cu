#ifndef _MAX_MEM_
#define _MAX_MEM_

// includes -------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "max_mem_kernel.cu"	// Kernel to Maximize FLOPS

// Defines --------------------------------------------------------------------
#define NUM_BLOCKS 				32768		// Keep the MPs busy for a while
#define NUM_THREADS_PER_BLOCK 384		//	Keep the cores occupied
#define BYTES_PER_FLOAT			4			// Duh...
#define OFFSET						0			// Test Coalescing
#define ARRAY_SIZE				(NUM_BLOCKS*NUM_THREADS_PER_BLOCK+OFFSET) 	// Combine


// Forward Declarations --------------------------------------------------------
void runTest( int argc, char** argv);
void init_arrays(float** h_in, float** h_out, float** d_in, float** d_out, size_t size);
void checkCUDAError(const char *msg);

// Main -----------------------------------------------------------------------
int main( int argc, char** argv) {
	runTest( argc, argv);

	exit(0);
}

// runTest --------------------------------------------------------------------
//		Runs a simple test to maximize the amount of memory bandwidth available on the GPU.
//
void runTest( int argc, char** argv) {

	// Hardware Dependent - NV GeForce 9500 GT
	unsigned int threads = NUM_BLOCKS * NUM_THREADS_PER_BLOCK;	
	
	printf("Number of Blocks:         %4d\n", NUM_BLOCKS);
	printf("Number of Threads/Blocks: %4d\n", NUM_THREADS_PER_BLOCK);
	printf("Number of Total Threads:  %4d\n", threads);
	printf("\n");
	

	// Initialize arrays on host and device
	printf("Init counters\n");
	float *h_in, *h_out, *d_in, *d_out;
	
	init_arrays(&h_in, &h_out, &d_in, &d_out, ARRAY_SIZE);

	// Create and Start Timer
	printf("Starting Test\n");
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );

	// Run the test
	max_flops_kernel<<< NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_in, d_out, OFFSET);

	// Record end time
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	float time_s = time/1000.0f;
	printf("Finished Test in %f s\n", time_s);

	// Check for errors
	checkCUDAError("test finished");
		
	// Check array
	// cudaMemcpy(h_out, d_out, threads * sizeof(float), cudaMemcpyDeviceToHost);
	// for(int i = 0; i < ARRAY_SIZE; ++i)
	// 	printf("A[%4d]: %.0f\n", i, h_out[i]);

	// Calculate GMEMOPS
	//	
	double total_bytes = ((double)threads * BYTES_PER_FLOAT * N_MEM_OPS_PER_KERNEL) / (1024.0*1024.0*1024.0) ;
	printf("Total GBytes Transferred: %.3f\n", total_bytes);
	double gbps = total_bytes / time_s;
	printf("GBps: %.3f\n", gbps);


	// Cleanup
	free(h_in);
	free(h_out);
	cudaFree(d_in);
	cudaFree(d_out);
	
}

// init_counters --------------------------------------------------------------
//		Initializes an array of floats that will be used to count FLOPS.
//
void init_arrays(float** h_in, float** h_out, float** d_in, float** d_out, size_t size){
	// Allocate host memory
	*h_in  = (float*) malloc( size * sizeof(float));
	*h_out = (float*) malloc( size * sizeof(float));	
	
	// Allocate device memory
	cudaMalloc((void **) d_in,  size * sizeof(float));
	cudaMalloc((void **) d_out, size * sizeof(float));
	checkCUDAError("malloc");	// Check for allocation errors
	
	// Initialize in arrays ...
	for( unsigned int i = 0; i < size; ++i)
		(*h_in)[i] = (float)i;
		
	// ... and copy to device
	cudaMemcpy(*d_in, *h_in, size * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError("memcpy"); // Check for initialization errors
	
}

//-----------------------------------------------------------

//From Dr Dobbs "CUDA: Supercomputing for the masses, Part 3"
//http://drdobbs.com/architecture-and-design/207200659      
//-----------------------------------------------------------

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, 
                             cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}


#endif /* MAX_FLOPS */
