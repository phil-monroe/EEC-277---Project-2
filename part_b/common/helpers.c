// Predeclarations ------------------------------------------------------------
void initArray(float** host, float** device, size_t size, float initial_value=0.0f);
void checkCUDAError(const char *msg);
void startTest(cudaEvent_t &start, cudaEvent_t &stop);
float finishTest(cudaEvent_t &start, cudaEvent_t &stop);


// Struct to hold the app's state
struct appConfig{
	unsigned int nBlocks;
	unsigned int nThreadsPerBlock;
	unsigned int nThreads;
	unsigned int nLoops;
	unsigned int arraySize;
} appConfig;


// displayHeader --------------------------------------------------------------
//		Print header message and common config variable
//    @param msg - Title for the test
// ----------------------------------------------------------------------------
void displayHeader(const char *msg){
	printf("%s \n", msg);
	printf("Written by Phil Monroe and Kramer Straube\n\n");
	
	printf("Number of Blocks       : %8d\n", appConfig.nBlocks);
	printf("Number of Threads/Block: %8d\n", appConfig.nThreadsPerBlock);
	printf("Number of Threads      : %8d\n", appConfig.nThreads);
	
	if(appConfig.nLoops)
		printf("Number of Loops        : %8d\n", appConfig.nLoops);
	printf("\n");
}
	
	
// initArray ------------------------------------------------------------------
//		Initializes an array of floats on the host and device
//    @param host				- The array on the host to initialize.
//		@param device			- The array on the device to initialize
//		@param size				- The length of the arrays to allocate.
//		@param initial_value	- The value to initialize the arrays to. -1.0 will
//										cause the array to initialize to the index value.
// ----------------------------------------------------------------------------
void initArray(float** host, float** device, size_t size, float initial_value){
	// Allocate host memory
	*host  = (float*) malloc( size * sizeof(float));

	// Allocate device memory
	cudaMalloc((void **) device, size * sizeof(float));
	checkCUDAError("malloc");	// Check for allocation errors

	// Initialize arrays ...
	for( size_t i = 0; i < size; ++i)
		(*host)[i] = initial_value == -1.0f ? i : initial_value;

	// ... and copy to device
	cudaMemcpy(*device, *host, size * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError("memcpy"); // Check for initialization errors
}


// checkCUDAError -------------------------------------------------------------
//		Convience method to check for cuda errors.
//		@param msg - Unique identifier to help debug.
//
//		From Dr Dobbs "CUDA: Supercomputing for the masses, Part 3"
//		http://drdobbs.com/architecture-and-design/207200659      
//-----------------------------------------------------------------------------
void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}


// startTest ------------------------------------------------------------------
//		Initializes the cuda timer events and starts the timer.
//		@param start - Start time evet
//		@param end   - End time evet
//-----------------------------------------------------------------------------
void startTest(cudaEvent_t &start, cudaEvent_t &stop){
	// Create Events
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	
	// Start Timer
	printf("Starting Test\n");
	cudaEventRecord( start, 0 );
}


// finishTest ------------------------------------------------------------------
//		Initializes the cuda timer events and starts the timer.
//		@param start - Start time evet
//		@param end   - End time evet
//		@returns the elapsed time in seconds.
//-----------------------------------------------------------------------------
float finishTest(cudaEvent_t &start, cudaEvent_t &stop){
	float time;
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	printf("Finished Test in %f s\n\n", time/1000.0f);
	
	// Check for errors
	checkCUDAError("test finished");
	
	// Return elapsed time
	return time/1000.0f;
}