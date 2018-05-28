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
	// which pixel does this thread have to work on?
	int column = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;

	if ((column < columns) && (row < rows)) {
		// calculate offset to access correct element
		int offset = (column)             // offset in a row
			+(columns * row);    // select row

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
				sobelValueX += inputImage[pixelOffset] * kernelX[index];
			}
			else {
				sobelValueX += inputImage[offset] * kernelX[index];
			}
		}

		// iterate all values in kernelY and 8 neighbours
		float sobelValueY = 0;
		for (int index = 0; index < 9; index++) {
			int pixelOffset = (column + pixelColumnOffsets[index]) + ((row + pixelRowOffsets[index]) * columns);
			if (pixelOffset >= 0 && pixelOffset < rows * columns) {
				sobelValueY += inputImage[pixelOffset] * kernelY[index];
			}
			else {
				sobelValueY += inputImage[offset] * kernelY[index];
			}
			
		}

		unsigned char sobelValue = sqrtf(sobelValueX * sobelValueX + sobelValueY * sobelValueY);
		outputImage[offset] = sobelValue;
	}
}
