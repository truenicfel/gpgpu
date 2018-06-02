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

__global__ void sobel(unsigned char* outputImage, unsigned char* inputImage, int rows, int columns)
{

	// shared memory (the second index accesses the row)
	__shared__ float ds_PIXELS[16][16];


	// which column and row does this thread have in the GRID?
	int columnGrid = blockIdx.x*blockDim.x + threadIdx.x;
	int rowGrid = blockIdx.y*blockDim.y + threadIdx.y;

	// check if this thread is in the area of the picture + 1 (is this thread active?)
	bool threadActive = columnGrid >= 0 && columnGrid <= columns + 1 && rowGrid >= 0 && rowGrid <= rows + 1;

	// check if this thread is in the area of the picture (not at the edges of the grid)
	bool inPicture = columnGrid > 0 && columnGrid <= columns && rowGrid > 0 && rowGrid <= rows;

	// which column and row does this thread have in the PICTURE?
	int columnPicture = blockIdx.x*blockDim.x + threadIdx.x - 1;
	int rowPicture = blockIdx.y*blockDim.y + threadIdx.y - 1;

	// calculate picture offset
	int offset = (columnPicture)+(columns * rowPicture);

	if (threadActive) {

		// load stuff into shared memory
		if (inPicture) {
			ds_PIXELS[blockDim.x][blockDim.y] = inputImage[offset];
		}
		else {
			ds_PIXELS[blockDim.x][blockDim.y] = 0;
		}
	}

	// wait until all threads finished loading
	__syncthreads();
	
	if (inPicture) {

		// the sobel kernels
		int kernelX[] = { 1, 0, -1, 2, 0, -2, 1, 0, -1 };
		int kernelY[] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };

		// iterate all values in kernelX and 8 neighbours
		float sobelValueX = 0;
		for (int index = 0; index < 9; index++) {
			sobelValueX += ds_PIXELS[threadIdx.y][threadIdx.x] * kernelX[8-index];
		}

		// iterate all values in kernelY and 8 neighbours
		float sobelValueY = 0;
		for (int index = 0; index < 9; index++) {
			sobelValueY += ds_PIXELS[threadIdx.y][threadIdx.x] * kernelY[8-index];
		}

		unsigned char sobelValue = sqrtf(sobelValueX * sobelValueX + sobelValueY * sobelValueY);
		outputImage[offset] = sobelValue;
	}
}
