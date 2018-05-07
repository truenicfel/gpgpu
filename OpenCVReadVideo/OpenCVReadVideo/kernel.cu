#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// convert the given color image to a grayscale image
// the input and output data is interpreted as follows:
// the data has number of rows given by "rows"
// the data has number of columns given by "columns"
// each element has size 3bytes each corresponding to one channel of the image b, g, r
__global__ void colorConvert(unsigned char* grayImage, unsigned char* colorImage, int rows, int columns)
{
	// which pixel does this thread have to work on?
	int column = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
  
	if ((column < columns) && (row < rows)) {
		// calculate offset to access correct element
		int offset = (column * 3)             // offset in a row
					+ (columns * 3 * row);    // select row
		// calculate grey values
		unsigned char greyValue = 0.07 * colorImage[offset] + 0.71 * colorImage[offset + 1] + 0.21 * colorImage[offset + 2];
		// copy one value to the result matrix and set other 2 to zero
		// first channel (blue)
		grayImage[offset] = greyValue;
		// second channel (green)
		grayImage[offset + 1] = greyValue;
		// third channel (red)
		grayImage[offset + 2] = greyValue;
	}
}
