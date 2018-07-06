#include "cudaKernel.h"
#include <helper_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda_gl_interop.h>

void cudaGetOpenCVImageSize(unsigned int &cols, unsigned int &rows) {
	cols = 640;
	rows = 360;
}

cudaGraphicsResource_t vboRes;
cudaGraphicsResource_t texRes;

void cudaInit ( unsigned int texId, unsigned int vboId, unsigned int cols, unsigned int rows){

	// Registration with CUDA.
	cudaGraphicsGLRegisterImage(&texRes, texId, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);
	cudaGraphicsGLRegisterBuffer(&vboRes, vboId, cudaGraphicsRegisterFlagsNone);
}

unsigned char *dev_mat_orig = NULL;
unsigned char *dev_histo = NULL;
cudaArray* texArray;
float*  vboArray;

__global__ void cudaUncachedHistogramKernel(unsigned char *input, size_t length, int step, float *out) {

	int i = blockIdx.x*blockDim.x*step + threadIdx.x*step;

	if (i < 256) {
		out[3 * i] = 0.0f;
	}
	
	__syncthreads();

	int num_threads = blockDim.x*gridDim.x;

	if (i < length) {
		if (input[i] < 256) {
			//atomicAdd((unsigned int*)&(out[input[i]]), 1);
			atomicAdd((float*)&(out[3*input[i]+1]), 1.0f);
		}
	}

	/*	for (int j = 0; j < 256 * 3; j++) {
		//if (j == 0) printf("sz=%d\n", length);
		out[j] = 0.0f;
	}
	out[4] = 0.1f;*/
}

#define TILE_WIDTH 32
int cudaExecOneStep(unsigned char *data, size_t sz, int step, int channels, int rows, int cols) {
	cudaError_t ret;

	if (dev_mat_orig == NULL) {
		ret = cudaMalloc((void**)&dev_mat_orig, sz);

		if (ret != cudaSuccess) {
			printf("cudaMemcpy() error %s\n", cudaGetErrorString(ret));
			exit(EXIT_FAILURE);
		}
	}

	/* copy original to device */
	ret = cudaMemcpy(dev_mat_orig, data, sz, cudaMemcpyHostToDevice);
	if (ret != cudaSuccess) {
		printf("cudaMemcpy() error %s\n", cudaGetErrorString(ret));
		exit(EXIT_FAILURE);
	}

	ret = cudaGraphicsMapResources(1, &texRes);
	if (ret != cudaSuccess) {
		printf("cudaMemcpy() error %s\n", cudaGetErrorString(ret));
		exit(EXIT_FAILURE);
	}
	ret = cudaGraphicsMapResources(1, &vboRes);	if (ret != cudaSuccess) {
		printf("cudaMemcpy() error %s\n", cudaGetErrorString(ret));
		exit(EXIT_FAILURE);
	}

	size_t vbosz = 0;

	ret = cudaGraphicsSubResourceGetMappedArray(&texArray, texRes, 0, 0);	if (ret != cudaSuccess) {
		printf("cudaMemcpy() error %s\n", cudaGetErrorString(ret));
		exit(EXIT_FAILURE);
	}	ret = cudaGraphicsResourceGetMappedPointer((void**)&vboArray, &vbosz, vboRes);	if (ret != cudaSuccess) {
		printf("cudaMemcpy() error %s\n", cudaGetErrorString(ret));
		exit(EXIT_FAILURE);
	}	//printf("r*c=%d\n", rows * cols);	// kernel aufruf hier:	// dev_mat_orig ist input	// texArray ist output	dim3 blocksPerGrid(ceil((input->rows * input->cols * 3) / TILE_WIDTH));
	dim3 threadsPerBlock(TILE_WIDTH);	cudaUncachedHistogramKernel << <ceil(1.0*(rows * cols) / TILE_WIDTH), TILE_WIDTH >> >(dev_mat_orig, sz, 1, vboArray);

	ret = cudaGetLastError();
	if (ret != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(ret));
		exit(EXIT_FAILURE);
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	ret = cudaDeviceSynchronize();
	if (ret != cudaSuccess) {
		fprintf(stderr, "HISTO: cudaDeviceSynchronize returned error code %d after launching addKernel! %s\n",
			ret, cudaGetErrorString(ret));
		exit(EXIT_FAILURE);
	}

	//cudaMemset(vboArray, 0, vbosz);

	cudaMemcpyToArray(texArray, 0, 0, dev_mat_orig, sz, cudaMemcpyDeviceToDevice);

	cudaGraphicsUnmapResources(1, &texRes);
	ret = cudaGraphicsUnmapResources(1, &vboRes);
	if (ret != cudaSuccess) {
		printf("cudaMemcpy() error %s\n", cudaGetErrorString(ret));
		exit(EXIT_FAILURE);
	}

	return 0;
}
