#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// convert the given color image to a grayscale image
// the input data is interpreted as follows:
// the data has number of rows given by "rows"
// the data has number of columns given by "columns"
// each element has size 3bytes each corresponding to one channel of the image b, g, r
// the output data is interpreted as follows:
// the data has number of rows given by "rows"
// the data has number of columns given by "columns"
// each element has size 1byte for one channel (the grayscale channel)
__global__ void colorConvert(unsigned char* grayImage, unsigned char* colorImage, int rows, int columns)
{
	// which pixel does this thread have to work on?
	int column = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
  
	if ((column < columns) && (row < rows)) {
		// calculate offset to access correct element
		int offset = (column)             // offset in a row
					+ (columns * row);    // select row
		// calculate grey values
		unsigned char grayValue = 0.07 * colorImage[offset * 3] + 0.71 * colorImage[offset*3 + 1] + 0.21 * colorImage[offset*3 + 2];
		// copy one value to the result matrix and set other 2 to zero
		// first channel (blue)
		grayImage[offset] = grayValue;
	}
}

// the gray image is a two dimensional array (row major) with # of rows given by rows and # number of columns given by columns
// the histogramm vector is the output of this kernel. its an array of ints (4 bytes) with size 256 (# of possible gray values)
__global__ void histogrammPrimitive(unsigned int* histogrammVector, unsigned char* grayImage, int rows, int columns)
{
	// which pixel does this thread have to work on?
	int column = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	
	// calculate offset to access correct element
	int offset = (column) + (columns * row);
	
	// check if im in scope of the picture
	if ((column < columns) && (row < rows)) {
		// load gray Value from input image
		unsigned char grayValue = grayImage[offset];
		// add up
		atomicAdd(&(histogrammVector[grayValue]), 1);
	}
}

// the input image is now interpreted as a long one dimensional vector of bytes. the size is given by size.
// the stride parameter tells this thread how many elements should be added up. the access to the memory is interleaved
__global__ void histogrammStride(unsigned int* histogrammVector, unsigned char* grayImage, int size)
{
	// stride is total number of threads
	int stride = blockDim.x * gridDim.x;

	// this is our start index in a stride (the element in a consecutive-elements-block that we have to add up)
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	// keep adding up until jumping out of the picture
	while (index < size) {
		// add up
		atomicAdd(&(histogrammVector[grayImage[index]]), 1);
		// step stride forward
		index += stride;
	}
}

// the input image is now interpreted as a long one dimensional vector of bytes. the size is given by size.
// the stride parameter tells this thread how many elements should be added up. the access to the memory is interleaved
// and this will use shared memory to add up the values locally
__global__ void histogrammStrideShared(unsigned int* histogrammVector, unsigned char* grayImage, int size)
{
	// shared memory to add up the pixel values
	__shared__ unsigned int histogram[256];

	// zero shared memory (blockDim must be smaller than 256)
	int toZeroPerThread = 256;
	int index = threadIdx.x;
	while (index < toZeroPerThread) {
		histogram[index] = 0;
		index += blockDim.x;
	}
	__syncthreads();

	// stride is total number of threads
	int stride = blockDim.x * gridDim.x;

	// this is our start index in a stride (the element in a consecutive-elements-block that we have to add up)
	index = threadIdx.x + blockIdx.x * blockDim.x;

	// keep adding up until jumping out of the picture
	while (index < size) {
		// add up (index to access correct element)
		atomicAdd(&(histogram[grayImage[index]]), 1);
		// step stride forward
		index += stride;
	}
	__syncthreads();

	// now all the shared memory block have to be added up in global memory (histogrammVector)
	// we can use the variable to Zeroper thread again
	int toAddPerThread = 256;
	index = threadIdx.x;	
	while (index < toAddPerThread) {
		atomicAdd(&(histogrammVector[index]), histogram[index]);
		index += blockDim.x;
	}
}

__global__ void sobel(unsigned char* outputImage, unsigned char* inputImage, int rows, int columns)
{

	// shared memory (the second index accesses the row)
	__shared__ unsigned char ds_PIXELS[16][16];

	// picture coordinates
	int column = blockIdx.x*(blockDim.x-2) + threadIdx.x - 1;
	int row = blockIdx.y*(blockDim.y-2) + threadIdx.y - 1;

	// check if this thread is in the area of the picture + 1 (is this thread active?)
	bool threadActive = column <= columns && row <= rows;

	// check if this thread is in the area of the picture (not at the edges of the grid)
	bool inPicture = column > 0 && column < columns && row > 0 && row < rows;

	// check if this thread is has to compute (true) or only load (false)
	bool hasToCompute = threadIdx.x > 0 && threadIdx.y > 0 && threadIdx.x < blockDim.x - 1 && threadIdx.y < blockDim.y - 1;

	// calculate picture offset
	int offset = (column) + (columns * row);

	if (threadActive) {

		// load stuff into shared memory
		if (inPicture) {
			ds_PIXELS[threadIdx.y][threadIdx.x] = inputImage[offset];
		}
		else {
			ds_PIXELS[threadIdx.y][threadIdx.x] = 0;
		}
	}

	// wait until all threads finished loading
	__syncthreads();


	if (hasToCompute && threadActive) {

		// the sobel kernels
		int kernelX[] = { 1, 0, -1, 2, 0, -2, 1, 0, -1 };
		int kernelY[] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };

		// the offsets for the columns to get the pixels
		int pixelColumnOffsets[] = {
			-1, 0,	1,
			-1, 0, 1,
			-1, 0,	1
		};
		
		// the offsets for the rows to get the pixels
		int pixelRowOffsets[] = {
			-1, -1,	-1,
			0,  0,  0,
			1,  1,  1
		};

		// iterate all values in kernelX and 8 neighbours
		float sobelValueX = 0;
		for (int index = 0; index < 9; index++) {
			sobelValueX += ds_PIXELS[threadIdx.y + pixelRowOffsets[index]][threadIdx.x + pixelColumnOffsets[index]] * kernelX[index];
		}

		// iterate all values in kernelY and 8 neighbours
		float sobelValueY = 0;
		for (int index = 0; index < 9; index++) {
			sobelValueY += ds_PIXELS[threadIdx.y + pixelRowOffsets[index]][threadIdx.x + pixelColumnOffsets[index]] * kernelY[index];
		}
		unsigned char sobelValue = sqrtf(sobelValueX * sobelValueX + sobelValueY * sobelValueY);
		
		outputImage[offset] = sobelValue;
	}
	
}

__global__ void sobelTexture(unsigned char* outputImage, cudaTextureObject_t inputImage, int rows, int columns)
{
	// which pixel does this thread have to work on?
	int column = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;

	if ((column < columns) && (row < rows)) {
		// calculate offset to access correct element
		int offset = (column)+(columns * row);

		// the sobel kernels
		int kernelX[] = { 1, 0, -1, 2, 0, -2, 1, 0, -1 };
		int kernelY[] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };

		// the offsets for the columns to get the pixels
		int pixelColumnOffsets[] = {
			-1, 0,	1,
			-1, 0, 1,
			-1, 0,	1
		};

		// the offsets for the rows to get the pixels
		int pixelRowOffsets[] = {
			-1, -1,	-1,
			0,  0,  0,
			1,  1,  1
		};

		// iterate all values in kernelX and 8 neighbours
		float sobelValueX = 0;
		for (int index = 0; index < 9; index++) {
			int pixelOffset = (column + pixelColumnOffsets[index]) + ((row + pixelRowOffsets[index]) * columns);
			if (pixelOffset >= 0 && pixelOffset < rows * columns) {
				sobelValueX += tex1D<float>(inputImage, pixelOffset) * kernelX[index];
			}
			else {
				sobelValueX += tex1D<float>(inputImage, offset) * kernelX[index];
			}
		}

		// iterate all values in kernelY and 8 neighbours
		float sobelValueY = 0;
		for (int index = 0; index < 9; index++) {
			int pixelOffset = (column + pixelColumnOffsets[index]) + ((row + pixelRowOffsets[index]) * columns);
			if (pixelOffset >= 0 && pixelOffset < rows * columns) {
				sobelValueY += tex1D<float>(inputImage, pixelOffset) * kernelY[index];
			}
			else {
				sobelValueY += tex1D<float>(inputImage, offset) * kernelY[index];
			}

		}

		unsigned char sobelValue = sqrtf(sobelValueX * sobelValueX + sobelValueY * sobelValueY);
		outputImage[offset] = sobelValue;
	}
}
