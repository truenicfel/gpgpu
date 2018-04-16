/*
 * Gruppe: 
 * Nico Daﬂler
 * Lukas Dorner
 * 
 * Datum:
 * 16.04.18
 */

#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_functions.h>

// macro for error checking

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() { cudaError_t error = cudaGetLastError(); if (error != cudaSuccess) { printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(error)); exit(EXIT_FAILURE);}}

void constantInit(float *data, int size, float val)
{
	for (int i = 0; i < size; ++i)
	{
		data[i] = val;
	}
}

/**
* Matrix multiplication (CUDA Kernel) on the device: C = A * B
* wA is A's width and wB is B's width
*/
template <int BLOCK_SIZE> __global__ void
matrixMulCUDA(float *C, float *A, float *B, int wA, int wB)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Index of the first sub-matrix of A processed by the block
	int aBegin = wA * BLOCK_SIZE * by;

	// Index of the last sub-matrix of A processed by the block
	int aEnd = aBegin + wA - 1;

	// Step size used to iterate through the sub-matrices of A
	int aStep = BLOCK_SIZE;

	// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * bx;

	// Step size used to iterate through the sub-matrices of B
	int bStep = BLOCK_SIZE * wB;

	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread
	float Csub = 0;

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int a = aBegin, b = bBegin;
	a <= aEnd;
		a += aStep, b += bStep)
	{

		// Declaration of the shared memory array As used to
		// store the sub-matrix of A
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

		// Declaration of the shared memory array Bs used to
		// store the sub-matrix of B
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		// Load the matrices from device memory
		// to shared memory; each thread loads
		// one element of each matrix
		As[ty][tx] = A[a + wA * ty + tx];
		Bs[ty][tx] = B[b + wB * ty + tx];

		// Synchronize to make sure the matrices are loaded
		__syncthreads();

		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix
#pragma unroll

		for (int k = 0; k < BLOCK_SIZE; ++k)
		{
			Csub += As[ty][k] * Bs[k][tx];
		}

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}

	// Write the block sub-matrix to device memory;
	// each thread writes one element
	int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	C[c + wB * ty + tx] = Csub;
}

/**
* Run a simple test of matrix multiplication using CUDA
*/
int matrixMultiply(int block_size, dim3 &dimsA, dim3 &dimsB)
{
	printf("Computing matrix multiplication on GPU using example code...\r\n");
	// Allocate host memory for matrices A and B
	unsigned int size_A = dimsA.x * dimsA.y;
	unsigned int mem_size_A = sizeof(float) * size_A;
	float *h_A = (float *)malloc(mem_size_A);
	unsigned int size_B = dimsB.x * dimsB.y;
	unsigned int mem_size_B = sizeof(float) * size_B;
	float *h_B = (float *)malloc(mem_size_B);

	// Initialize host memory
	const float valB = 0.01f;
	constantInit(h_A, size_A, 1.0f);
	constantInit(h_B, size_B, valB);

	// Allocate device memory
	float *d_A, *d_B, *d_C;

	// Allocate host matrix C
	dim3 dimsC(dimsB.x, dimsA.y, 1);
	unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
	float *h_C = (float *)malloc(mem_size_C);

	if (h_C == NULL)
	{
		fprintf(stderr, "Failed to allocate host matrix C!\n");
		exit(EXIT_FAILURE);
	}

	cudaError_t error;

	error = cudaMalloc((void **)&d_A, mem_size_A);

	if (error != cudaSuccess)
	{
		printf("cudaMalloc d_A returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	error = cudaMalloc((void **)&d_B, mem_size_B);

	if (error != cudaSuccess)
	{
		printf("cudaMalloc d_B returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	error = cudaMalloc((void **)&d_C, mem_size_C);

	if (error != cudaSuccess)
	{
		printf("cudaMalloc d_C returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	// copy host memory to device
	error = cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);

	if (error != cudaSuccess)
	{
		printf("cudaMemcpy (d_A,h_A) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	error = cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

	if (error != cudaSuccess)
	{
		printf("cudaMemcpy (d_B,h_B) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	// Setup execution parameters
	dim3 threads(block_size, block_size);
	dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

	// Performs warmup operation using matrixMul CUDA kernel
	if (block_size == 16)
	{
		matrixMulCUDA<16> << < grid, threads >> >(d_C, d_A, d_B, dimsA.x, dimsB.x);
	}
	else
	{
		matrixMulCUDA<32> << < grid, threads >> >(d_C, d_A, d_B, dimsA.x, dimsB.x);
	}
	return 0;
}

__global__ void MultKernel(float *c, float *a, float *b, int matrixSize)
{
	// get row/column
	int row = blockIdx.x*blockDim.x + threadIdx.x;
	int column = blockIdx.y*blockDim.y + threadIdx.y;


	// result for the chose element (given by row and column)
	float result = 0;
	// iterate over both chosen row/line and column
	for (int element = 0; element < 2; element++) {
		result += a[row * matrixSize + element] * b[element * matrixSize + column];
	}
	c[row * matrixSize + column] = result;

}

int matMultHost(int matrixSize) {
	printf("Computing matrix multiplication on CPU...\r\n");

	// matrix a
	// -----------------------------------------------------------------------------------------------------------------
	float* a;
	a = (float*)malloc(matrixSize * matrixSize * sizeof(float));

	if (a == NULL) {
		printf("malloc failed for matrix a.");
		exit(1);
	}

	// fill matrix a with values of counter which get incremented with every loop
	float counter = 0;
	for (int rowNumber = 0; rowNumber < matrixSize; rowNumber++) {
		for (int index = 0; index < matrixSize; index++) {
			a[rowNumber * matrixSize + index] = counter;
			counter++;
		}
	}
	// for testing of the algorithm
	//printf("Input Matrix A:\r\n");
	//printMatrix(a, matrixSize);

	// matrix b
	// -----------------------------------------------------------------------------------------------------------------
	float* b;
	b = (float*)malloc(matrixSize * matrixSize * sizeof(float));

	if (b == NULL) {
		printf("malloc failed for matrix b.");
		exit(1);
	}

	// fill matrix a with values of counter which get incremented with every loop
	for (int rowNumber = 0; rowNumber < matrixSize; rowNumber++) {
		for (int index = 0; index < matrixSize; index++) {
			b[rowNumber * matrixSize + index] = counter;
			counter--;
		}
	}
	// for testing of the algorithm
	//printf("Input Matrix B:\r\n");
	//printMatrix(b, matrixSize);

	// multiply
	// -----------------------------------------------------------------------------------------------------------------
	// result matrix c
	float* c;
	c = (float*)malloc(matrixSize * matrixSize * sizeof(float));

	if (c == NULL) {
		printf("malloc failed for matrix c.");
		exit(1);
	}

	// choose row/line
	for (int rowNumber = 0; rowNumber < matrixSize; rowNumber++) {
		// choose column
		for (int columnNumber = 0; columnNumber < matrixSize; columnNumber++) {
			// result for the chose element (given by row and column)
			float result = 0;
			// iterate over both chosen row/line and column
			for (int element = 0; element < matrixSize; element++) {
				result += a[rowNumber * matrixSize + element] * b[element * matrixSize + columnNumber];
			}
			c[rowNumber * matrixSize + columnNumber] = result;
		}
	}
	// for testing of the algorithm
	//printf("Result Matrix C:\r\n");
	//printMatrix(c, matrixSize);
	return 0;
}

int matMultCUDA(int matrixSize) {
	printf("Computing matrix multiplication on GPU...\r\n");

	float *d_A, *d_B, *d_C;
	float *h_A, *h_B, *h_C;

	// matrix a (host)
	// -----------------------------------------------------------------------------------------------------------------
	h_A = (float*)malloc(matrixSize * matrixSize * sizeof(float));

	if (h_A == NULL) {
		printf("malloc failed for matrix a.");
		exit(EXIT_FAILURE);;
	}

	// fill matrix a with values of counter which get incremented with every loop
	float counter = 0;
	for (int rowNumber = 0; rowNumber < matrixSize; rowNumber++) {
		for (int index = 0; index < matrixSize; index++) {
			h_A[rowNumber * matrixSize + index] = counter;
			counter++;
		}
	}
	// for testing of the multiplication
	//printf("Input Matrix A:\r\n");
	//printMatrix(h_A, matrixSize);

	// matrix b (host)
	// -----------------------------------------------------------------------------------------------------------------
	h_B = (float*)malloc(matrixSize * matrixSize * sizeof(float));

	if (h_B == NULL) {
		printf("malloc failed for matrix b.");
		exit(EXIT_FAILURE);
	}

	// fill matrix a with values of counter which get incremented with every loop
	for (int rowNumber = 0; rowNumber < matrixSize; rowNumber++) {
		for (int index = 0; index < matrixSize; index++) {
			h_B[rowNumber * matrixSize + index] = counter;
			counter--;
		}
	}
	// for testing of the algorithm
	//printf("Input Matrix B:\r\n");
	//printMatrix(h_B, matrixSize);

	// result matrix c (host)
	// -----------------------------------------------------------------------------------------------------------------
	h_C = (float*)malloc(matrixSize * matrixSize * sizeof(float));

	if (h_C == NULL) {
		printf("malloc failed for matrix c.");
		exit(EXIT_FAILURE);
	}

	// matrix a (device)
	// -----------------------------------------------------------------------------------------------------------------

	cudaMalloc((void **)&d_A, matrixSize * matrixSize *  sizeof(float));
	cudaCheckError();

	cudaMemcpy(d_A, h_A, matrixSize * matrixSize *  sizeof(float), cudaMemcpyHostToDevice);
	cudaCheckError();

	// matrix b (device)
	// -----------------------------------------------------------------------------------------------------------------

	cudaMalloc((void **)&d_B, matrixSize * matrixSize *  sizeof(float));
	cudaCheckError();

	cudaMemcpy(d_B, h_B, matrixSize * matrixSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaCheckError();

	// matrix c (device)
	// -----------------------------------------------------------------------------------------------------------------

	cudaMalloc((void **)&d_C, matrixSize * matrixSize * sizeof(float));
	cudaCheckError();

	// computation on device
	// -----------------------------------------------------------------------------------------------------------------

	
	dim3 gridDimensions;
	gridDimensions.x = 4;
	gridDimensions.y = 4;
	gridDimensions.z = 1;
	dim3 blockDimension;
	blockDimension.x = matrixSize/4;
	blockDimension.y = matrixSize/4;
	blockDimension.z = 1;
	MultKernel<<<gridDimensions, blockDimension>>>(d_C, d_A, d_B, matrixSize);
	cudaGetLastError();
	cudaCheckError();

	// copy result back to host
	// -----------------------------------------------------------------------------------------------------------------
	cudaMemcpy(h_C, d_C, matrixSize * matrixSize * sizeof(float), cudaMemcpyDeviceToHost);
	cudaCheckError();

	// free memory on device
	// -----------------------------------------------------------------------------------------------------------------
	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
	cudaCheckError();

	// for testing of the algorithm
	//printf("Result Matrix C:\r\n");
	//printMatrix(h_C, 2);

	return 0;
}


int main()
{
	StopWatchInterface *timer;

	if (!sdkCreateTimer(&timer)) {
		printf("Create timer failed!\n");
		exit(EXIT_FAILURE);
	}

	sdkStartTimer(&timer);
	// this gets measured
	matMultHost(2048);
	sdkStopTimer(&timer);
	printf("Zeitdauer (CPU): %f\n", sdkGetTimerValue(&timer));
	
	// new timer...
	if (!sdkCreateTimer(&timer)) {
		printf("Create timer failed!\n");
		exit(EXIT_FAILURE);
	}
	sdkStartTimer(&timer);
	// this gets measured
	matMultCUDA(2048);
	sdkStopTimer(&timer);
	printf("Zeitdauer (GPU): %f\n", sdkGetTimerValue(&timer));


	int block_size = 32;
	dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
	dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);

	dimsA.x = 2048;
	dimsA.y = 2048;
	dimsB.x = 2048;
	dimsB.y = 2048;

	// new timer...
	if (!sdkCreateTimer(&timer)) {
		printf("Create timer failed!\n");
		exit(EXIT_FAILURE);
	}
	sdkStartTimer(&timer);
	// this gets measured
	matrixMultiply(block_size, dimsA, dimsB);
	sdkStopTimer(&timer);
	printf("Zeitdauer (Exampole on GPUG): %f\n", sdkGetTimerValue(&timer));
	
	return 0;
}

