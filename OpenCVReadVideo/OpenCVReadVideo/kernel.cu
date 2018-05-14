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

		// the offsets for the offset
		int offsets[] = {
			- columns - 1,	- columns,	- columns + 1,
			- 1,			0,			1,		
			columns - 1,	columns,	columns + 1		
		};

		// iterate all values in kernelX and 8 neighbours
		unsigned char sobelValueX = 0;
		for (int index = 0; index < 9; index++) {
			int actualOffset = offset + offsets[index];
			if (actualOffset >= 0) {
				sobelValueX += inputImage[actualOffset] * kernelX[index];
			}
		}

		// iterate all values in kernelY and 8 neighbours
		unsigned char sobelValueY = 0;
		for (int index = 0; index < 9; index++) {
			int actualOffset = offset + offsets[index];
			if (actualOffset >= 0) {
				sobelValueY += inputImage[actualOffset] * kernelY[index];
			}
		}

		unsigned char sobelValue = (unsigned char) floor(sqrtf(pow((float)sobelValueX, 2.f) + pow((float)sobelValueY, 2.f))); 
		outputImage[offset] = sobelValue;
	}
}
