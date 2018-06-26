// OpenCVReadVideo.cpp : Definiert den Einstiegspunkt fuer die Konsolenanwendung.
//

#include <iostream>
#include "stdafx.h"

#include "opencv2/opencv.hpp"
#include "cuda_runtime.h"

using namespace cv;
using namespace std;

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() { cudaError_t error = cudaGetLastError(); if (error != cudaSuccess) { printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(error)); exit(EXIT_FAILURE);}}

extern void colorConvert(unsigned char* grayImage, unsigned char* colorImage, int rows, int columns);

extern void sobel(unsigned char* outputImage, unsigned char* inputImage, int rows, int columns);

extern void histogrammPrimitive(unsigned int* histogrammVector, unsigned char* grayImage, int rows, int columns);

extern void histogrammStride(unsigned int* histogrammVector, unsigned char* grayImage, int size);

extern void histogrammStrideShared(unsigned int* histogrammVector, unsigned char* grayImage, int size);

extern void sobelTexture(unsigned char* outputImage, cudaTextureObject_t inputImage, int rows, int columns);

// provide with a frame the size of the frame: columns = widthofFrame and rows = height of frame
Mat modifyFrame(Mat frame, bool print)
{
	int rows = frame.rows;
	int columns = frame.cols;

	// this is fixed for now: 16 x 16 block size
	int blockSize = 16;
	dim3 blockDimension = dim3(blockSize, blockSize, 1);
	dim3 gridDimension = dim3((frame.cols - 1) / (blockSize) + 1, (frame.rows - 1) / (blockSize) + 1, 1);

	// size of mat data
	size_t frameDataSize = frame.elemSize() * static_cast<size_t>(frame.size[0]) * static_cast<size_t>(frame.size[1]) * sizeof(uint8_t);
	size_t grayFrameDataSize = frameDataSize / 3;
	size_t histogrammSize = sizeof(int) * 256;
	size_t totalNumberOfPixels = rows * columns;

	// device pointer
	uchar* device_input = nullptr;
	uchar* device_output = nullptr;
	uchar* device_result = nullptr;
	unsigned int* device_histogramm = nullptr;

	// malloc for input
	cudaMalloc((void**)&device_input, frameDataSize);
	cudaCheckError();
	// malloc for output (this is 3 times smaller since we only store gray values and not bgr)
	cudaMalloc((void**)&device_output, grayFrameDataSize);
	cudaCheckError();
	// malloc for result (this will store the result of sobel kernel)
	cudaMalloc((void**)&device_result, grayFrameDataSize);
	cudaCheckError();
	// malloc for histogramm result (this will store the result of histogramm kernel)
	cudaMalloc((void**)&device_histogramm, histogrammSize);
	cudaCheckError();

	// now copy the actual data to the device as input for the kernel
	cudaMemcpy(device_input, frame.data, frameDataSize, cudaMemcpyHostToDevice);
	cudaCheckError();

	// launch the color convert which converts to grayscale
	void* args1[] = { &device_output, &device_input, &rows, &columns };
	cudaLaunchKernel<void>(&colorConvert, gridDimension, blockDimension, args1);
	cudaCheckError();

	// adapt the the dimensions
	blockDimension = dim3(blockSize, 1, 1);
	gridDimension = dim3(((frame.cols * frame.rows) / (blockDimension.x * 64)), 1, 1);

	// now launch the histogramm kernel
	void* args2[] = { &device_histogramm, &device_output, &totalNumberOfPixels};
	cudaLaunchKernel<void>(&histogrammStrideShared, gridDimension, blockDimension, args2);
	cudaCheckError();

	Mat result(rows, columns, CV_8UC1);

	// write modified data to result
	cudaMemcpy(result.data, device_output, grayFrameDataSize, cudaMemcpyDeviceToHost);
	cudaCheckError();

	int histogramm[256] = { 0 };

	cudaMemcpy(histogramm, device_histogramm, histogrammSize, cudaMemcpyDeviceToHost);
	cudaCheckError();

	if (print) {
		int sum = 0;
		for (int i = 0; i < 256; i++) {
			sum += histogramm[i];
		}
		std::cout << sum << std::endl;
	}

	cudaFree(device_input);
	cudaCheckError();
	cudaFree(device_output);
	cudaCheckError();
	cudaFree(device_result);
	cudaCheckError();
	cudaFree(device_histogramm);
	cudaCheckError();
	return result;
}

Mat grayAndSobelUsingStream(Mat frame) {

	int rows = frame.rows;
	int columns = frame.cols;

	// this will also be used as an intermediate storage
	Mat result(rows, columns, CV_8UC1);

	// this is fixed for now: 16 x 16 block size
	int blockSize = 16;
	dim3 blockDimension = dim3(blockSize, blockSize, 1);
	dim3 gridDimensionColorConvert = dim3((frame.cols - 1) / (blockSize) + 1, (frame.rows - 1) / (blockSize) + 1, 1);
	dim3 gridDimensionSobel = dim3((frame.cols - 1) / (blockSize - 2) + 1, (frame.rows - 1) / (blockSize - 2) + 1, 1);

	// size of mat data
	size_t frameDataSize = frame.elemSize() * static_cast<size_t>(frame.size[0]) * static_cast<size_t>(frame.size[1]) * sizeof(uint8_t);
	size_t grayFrameDataSize = frameDataSize / 3;

	// device pointer
	uchar* device_input = nullptr;
	uchar* device_output = nullptr;

	// host alloc for input (pinned memory)
	cudaHostAlloc((void**)&device_input, frameDataSize, cudaHostAllocDefault);
	cudaCheckError();
	// host alloc for output (pinned memory)(this is 3 times smaller since we only store gray values and not bgr)
	cudaHostAlloc((void**)&device_output, grayFrameDataSize, cudaHostAllocDefault);
	cudaCheckError();

	// create streams for colorConvert- and sobel-kernel
	cudaStream_t streamColorConvert;
	cudaStream_t streamSobel;
	cudaStreamCreate(&streamColorConvert);
	cudaCheckError();
	cudaStreamCreate(&streamSobel);
	cudaCheckError();

	// create event that marks when the color convert stream is finished
	// (this is just testing stuff. this could be done easier using stream sync.)
	cudaEvent_t colorConvertFinished;
	cudaEventCreate(&colorConvertFinished);

	// fill the streams

	// color convert:
	// copy data as input for the kernel
	cudaMemcpyAsync(device_input, frame.data, frameDataSize, cudaMemcpyHostToDevice, streamColorConvert);
	cudaCheckError();

	// launch the color convert which converts to grayscale
	void* args1[] = { &device_output, &device_input, &rows, &columns };
	cudaLaunchKernel<void>(&colorConvert, gridDimensionColorConvert, blockDimension, args1, 0, streamColorConvert);
	cudaCheckError();

	cudaMemcpyAsync(result.data, device_output, grayFrameDataSize, cudaMemcpyDeviceToHost, streamColorConvert);
	cudaCheckError();

	// the next two instructions are just there for experimenting with streams

	// set event
	cudaEventRecord(colorConvertFinished, streamColorConvert);
	cudaCheckError();

	// sobel:
	cudaStreamWaitEvent(streamSobel, colorConvertFinished, 0);
	cudaCheckError();

	cudaMemcpyAsync(device_input, result.data, grayFrameDataSize, cudaMemcpyHostToDevice, streamSobel);
	cudaCheckError();

	// output becomes input and input becomes output
	void* args2[] = { &device_output, &device_input, &rows, &columns };
	cudaLaunchKernel<void>(&sobel, gridDimensionSobel, blockDimension, args2, 0, streamSobel);
	cudaCheckError();

	//// because device_input contains the result we will copy it from there
	cudaMemcpyAsync(result.data, device_output, grayFrameDataSize, cudaMemcpyDeviceToHost, streamSobel);
	cudaCheckError();

	// alternative:
	cudaDeviceSynchronize();
	//cudaStreamSynchronize(streamSobel);
	cudaCheckError();

	return result;

}

Mat sobelCuda(Mat frame) {

	int rows = frame.rows;
	int columns = frame.cols;

	// this is fixed for now: 16 x 16 block size
	int blockSize = 16;
	dim3 blockDimension = dim3(blockSize, blockSize, 1);
	dim3 gridDimension = dim3((frame.cols - 1) / (blockSize - 2) + 1, (frame.rows - 1) / (blockSize - 2) + 1, 1);

	// size of mat data
	size_t grayFrameDataSize = static_cast<size_t>(frame.size[0]) * static_cast<size_t>(frame.size[1]) * sizeof(uint8_t);

	// device pointer
	uchar* device_input = nullptr;
	uchar* device_output = nullptr;

	// malloc for input
	cudaMalloc((void**)&device_input, grayFrameDataSize);
	cudaCheckError();

	// malloc for output
	cudaMalloc((void**)&device_output, grayFrameDataSize);
	cudaCheckError();

	// now copy the actual data to the device as input for the kernel
	cudaMemcpy(device_input, frame.data, grayFrameDataSize, cudaMemcpyHostToDevice);
	cudaCheckError();

	// launch the sobel kernel
	void* args1[] = { &device_output, &device_input, &rows, &columns };
	cudaLaunchKernel<void>(&sobel, gridDimension, blockDimension, args1);
	cudaCheckError();

	Mat result(rows, columns, CV_8UC1);

	// write modified data to result
	cudaMemcpy(result.data, device_output, grayFrameDataSize, cudaMemcpyDeviceToHost);
	cudaCheckError();

	cudaFree(device_input);
	cudaCheckError();
	cudaFree(device_output);
	cudaCheckError();

	return result;
}

Mat sobelTextureCuda(Mat frame) {

	int rows = frame.rows;
	int columns = frame.cols;

	// this is fixed for now: 16 x 16 block size
	int blockSize = 16;
	dim3 blockDimension = dim3(blockSize, blockSize, 1);
	dim3 gridDimension = dim3((frame.cols - 1) / (blockSize - 2) + 1, (frame.rows - 1) / (blockSize - 2) + 1, 1);

	// size of mat data
	size_t grayFrameDataSize = static_cast<size_t>(frame.size[0]) * static_cast<size_t>(frame.size[1]) * sizeof(uint8_t);

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaArray_t cuArray;
	cudaMallocArray(&cuArray, &channelDesc, frame.size[0], frame.size[1]);
	cudaMemcpyToArray(cuArray, 0, 0, frame.data, grayFrameDataSize, cudaMemcpyHostToDevice);

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

	// device pointer
	uchar* device_input = nullptr;
	uchar* device_output = nullptr;

	// malloc for input
	cudaMalloc((void**)&device_input, grayFrameDataSize);
	cudaCheckError();

	// malloc for output
	cudaMalloc((void**)&device_output, grayFrameDataSize);
	cudaCheckError();

	// now copy the actual data to the device as input for the kernel
	cudaMemcpy(device_input, frame.data, grayFrameDataSize, cudaMemcpyHostToDevice);
	cudaCheckError();

	// launch the color convert which converts to grayscale
	void* args1[] = { &device_output, &texObj, &rows, &columns };
	cudaLaunchKernel<void>(&sobel, gridDimension, blockDimension, args1);
	cudaCheckError();

	Mat result(rows, columns, CV_8UC1);

	// write modified data to result
	cudaMemcpy(result.data, device_output, grayFrameDataSize, cudaMemcpyDeviceToHost);
	cudaCheckError();

	cudaFree(device_input);
	cudaCheckError();
	cudaFree(device_output);
	cudaCheckError();
	cudaDestroyTextureObject(texObj);
	cudaFreeArray(cuArray);
	cudaCheckError();

	return result;
}

void normalLaunch(VideoCapture cap) {
	Mat edges;
	namedWindow("edges", 1);

	// stores a single frame (input to device)
	Mat frame;
	// stores a single frame (output from device)
	Mat output;
	// get a new frame from camera
	cap >> frame;
	// print some video data...
	cout << "frame: dims: " << frame.dims << ", size[0]: " << frame.size[0] << ", size[1]:" << frame.size[1] << ", step[0]: " << frame.step[0] << ", step[1]:" << frame.step[1];
	cout << ", type: " << frame.type() << " (CV16U: " << CV_16UC1 << ", CV8UC3: " << CV_8UC3 << ")" << ", elemSize: " << frame.elemSize();
	cout << ", rows: " << frame.rows << ", cols: " << frame.cols << ", size: " << frame.size << ", dataPtr: " << frame.data << endl;
	int frameCounter = 0;
	for (;;)
	{

		if (frame.dims == 0) { // we're done
			break;
		}

		//output = modifyFrame(frame, frameCounter % 100 == 0);
		output = grayAndSobelUsingStream(frame);

		// ------------------------------------------------
		//cvtColor(frame, edges, COLOR_BGR2GRAY);
		//GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
		//Sobel(output, edges, frame.depth(), 2, 2);
		//Canny(edges, edges, 0, 30, 3);
		//imshow("edges", edges);
		// ------------------------------------------------

		// show the output from device
		imshow("edges", output);
		if (waitKey(1) >= 0) break;

		cap >> frame;
		frameCounter++;
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	getchar();
}

int main(int, char**)
{

	//VideoCapture cap("H:\\Benutzer\\gpgpu\\OpenCVReadVideo\\Videos\\Wildlife.wmv");
	VideoCapture cap("H:\\Benutzer\\Dokumente\\GPGPU\\gpgpu\\OpenCVReadVideo\\Videos\\robotica_1080.mp4");

	if (!cap.isOpened())  // check if we succeeded
		return -1;

	normalLaunch(cap);

	return 0;
}
