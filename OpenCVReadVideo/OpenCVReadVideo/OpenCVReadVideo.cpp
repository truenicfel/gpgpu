// OpenCVReadVideo.cpp : Definiert den Einstiegspunkt f�r die Konsolenanwendung.
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
	//printf("%p", (void*)device_input);
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

//Mat sobel(Mat frame) {
//
//	int rows = frame.rows;
//	int columns = frame.cols;
//
//	// this is fixed for now: 16 x 16 block size
//	int blockSize = 16;
//	dim3 blockDimension = dim3(blockSize, blockSize, 1);
//	dim3 gridDimension = dim3((frame.cols - 1) / (blockSize)+1, (frame.rows - 1) / (blockSize)+1, 1);
//
//	// size of mat data
//	size_t grayFrameDataSize = static_cast<size_t>(frame.size[0]) * static_cast<size_t>(frame.size[1]) * sizeof(uint8_t);
//
//	// device pointer
//	uchar* device_input = nullptr;
//	uchar* device_output = nullptr;
//
//	// malloc for input
//	cudaMalloc((void**)&device_input, grayFrameDataSize);
//	cudaCheckError();
//
//	// malloc for output 
//	cudaMalloc((void**)&device_output, grayFrameDataSize);
//	cudaCheckError();
//
//	// now copy the actual data to the device as input for the kernel
//	cudaMemcpy(device_input, frame.data, grayFrameDataSize, cudaMemcpyHostToDevice);
//	cudaCheckError();
//
//	// launch the color convert which converts to grayscale
//	void* args1[] = { &device_output, &device_input, &rows, &columns };
//	cudaLaunchKernel<void>(&sobel, gridDimension, blockDimension, args1);
//	cudaCheckError();
//
//}

int main(int, char**)
{

	//VideoCapture cap("H:\\Benutzer\\gpgpu\\OpenCVReadVideo\\Videos\\Wildlife.wmv");
	VideoCapture cap("H:\\Benutzer\\Dokumente\\GPGPU\\gpgpu\\OpenCVReadVideo\\Videos\\robotica_1080.mp4");

	if (!cap.isOpened())  // check if we succeeded
		return -1;

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

		output = modifyFrame(frame, frameCounter % 100 == 0);

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
	return 0;
}