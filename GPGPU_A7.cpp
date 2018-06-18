__global__ void colorConvert(unsigned char* grayImage, cudaTextureObject_t  colorImage, int rows, int columns)
{
	// which pixel does this thread have to work on?
	int column = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
  
	if ((column < columns) && (row < rows)) {
		// calculate offset to access correct element
		int offset = (column)             // offset in a row
					+ (columns * row);    // select row
		// calculate grey values
		//unsigned char grayValue = 0.07 * colorImage[offset * 3] + 0.71 * colorImage[offset*3 + 1] + 0.21 * colorImage[offset*3 + 2];
		
		unsigned char grayVal = 0.07 * tex1D<float>(colorImage, offset*3)
								+ 0.71 * tex1D<float>(colorImage, offset*3 + 1)
								+ 0.21 * tex1D<float>(colorImage, offset*3 + 2);
		// copy one value to the result matrix and set other 2 to zero
		// first channel (blue)
		grayImage[offset] = grayValue;
	}
}




cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindFloat);
cudaArray_t cuArray;
cudaMallocArray(&cuArray, &channelDesc, width, height);
cudaMemcpyToArray(cuArray, 0, 0, h_data, size, cudaMemcpyHostToDevice);

struct cudaResourceDesc resDesc;
memset(&resDesc, 0, sizeof(resDesc));
resDesc.resType = cudaResourceTypeArray;
resDesc.res.array.array = cuArray;

struct cudaTextureDesc texDesc;
memset(&texDesc, 0, sizeof(texDesc));
texDesc.addressMode[0] = cudaAddressModeWrap;
texDesc.addressMode[1] = cudaAddressModeWrap;
texDesc.filterMode = cudaFilterModeLinear;
texDesc.readMode = cudaReadModeElementType;
texDesc.normalizedCoords = 1;

cudaTextureObject_t texObj = 0;
cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

cudaDestroyTextureObject(texObj);
cudaFreeArray(cuArray);