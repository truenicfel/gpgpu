#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void addKernel(char* outputMatrix, char* inputMatrix, int rows, int columns)
{
  // which pixel does this thread have to work on?
	int column = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  
  if ((column < columns) && (row < rows)) {
    // each element has size 3 bytes (provide as param?)
    // calculate offset to access correct element
    int offset = (column * 3)             // offset in a row
                + (columns * 3 * row);    // select row
    // copy one value to the result matrix and set other 2 to zero
    // first channel (r, g or b? would be interesting to know)
    outputMatrix[offset] = inputMatrix[offset];
    // second channel
    outputMatrix[offset + 1] = 0;
    // third channel
    outputMatrix[offset + 2] = 0;
  }
}
