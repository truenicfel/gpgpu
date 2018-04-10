#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_functions.h>

// macro for error checking

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() { cudaError_t error = cudaGetLastError(); if (error != cudaSuccess) { printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(error)); exit(EXIT_FAILURE);}}

__global__ void MultKernel(float *c, float *a, float *b, int matrixSize)
{
	// get row/column
	int row = blockIdx.x;
	int column = blockIdx.y;


	// result for the chose element (given by row and column)
	float result = 0;
	// iterate over both chosen row/line and column
	for (int element = 0; element < 2; element++) {
		result += a[row * matrixSize + element] * b[element * matrixSize + column];
	}
	c[row * matrixSize + column] = result;

}

void printMatrix(float* pointerToMatrix, int matrixSize) {
	for (int rowNumber = 0; rowNumber < matrixSize; rowNumber++) {
		for (int index = 0; index < matrixSize; index++) {
			printf("%f  ", pointerToMatrix[rowNumber * matrixSize + index]);
		}
		printf("\r\n");
	}
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

	
	dim3 blockDimensions;
	blockDimensions.x = matrixSize;
	blockDimensions.y = matrixSize;
	MultKernel<<<blockDimensions, 1>>>(d_C, d_A, d_B, matrixSize);
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

	return 0;
}

