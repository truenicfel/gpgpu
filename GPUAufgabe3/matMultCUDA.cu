#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_functions.h>


cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

void printMatrix(float* pointerToMatrix, int matrixSize) {
	for (int rowNumber = 0; rowNumber < matrixSize; rowNumber++) {
		for (int index = 0; index < matrixSize; index++) {
			printf("%f  ", pointerToMatrix[rowNumber * matrixSize + index]);
		}
		printf("\r\n");
	}
}

int matMultHost() {
	printf("Hello, World!\n");

	int matrixSize = 2;

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
	printMatrix(a, matrixSize);

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
	printMatrix(b, matrixSize);

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
	printMatrix(c, matrixSize);
	return 0;
}

int matMultCUDA() {
	printf("Hello, World!\n");

	int matrixSize = 2;

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
	printMatrix(a, matrixSize);

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
	printMatrix(b, matrixSize);

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
	printMatrix(c, matrixSize);
	return 0;
}


int main()
{
	StopWatchInterface *t;
	if (!sdkCreateTimer(&t)) {
		printf("timercreate failed\n");
		exit(-1);
	}

	sdkStartTimer(&t);
	// zu vermessende Funktionalität
	sdkStopTimer(&t);

	printf("Zeitdauer: %f\n", sdkGetTimerValue(&t);

	matMultHost();
	matMultCUDA();

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <1, size >> >(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
