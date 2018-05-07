// OpenCVReadVideo.cpp : Definiert den Einstiegspunkt für die Konsolenanwendung.
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

// provide with a frame the size of the frame: columns = widthofFrame and rows = height of frame
Mat modifyFrame(Mat frame)
{
	
	// this is fixed for now: 16 x 16 block size
	int blockSize = 16;
	dim3 blockDimension = dim3(blockSize, blockSize, 1);
	dim3 gridDimension = dim3((frame.cols - 1) / blockSize + 1, (frame.rows - 1) / blockSize + 1, 1);

	// size of mat data
	size_t frameDataSize = frame.elemSize() * static_cast<size_t>(frame.size[0]) * static_cast<size_t>(frame.size[1]) * sizeof(uint8_t);

	// device pointer
	uchar* device_input = nullptr;
	uchar* device_output = nullptr;

	// malloc for input
	cudaMalloc((void**)&device_input, frameDataSize);
	cudaCheckError();
	//printf("%p", (void*)device_input);
	// malloc for output
	cudaMalloc((void**)&device_output, frameDataSize);
	cudaCheckError();

	// now copy the actual data to the device as input for the kernel
	cudaMemcpy(device_input, frame.data, frameDataSize, cudaMemcpyHostToDevice);
	cudaCheckError();

	int rows = frame.rows;
	int columns = frame.cols;

	void* args1[] = { &device_output, &device_input, &rows, &columns };
	cudaLaunchKernel<void>(&colorConvert, gridDimension, blockDimension, args1);
	cudaCheckError();
	cudaDeviceSynchronize();

	// clone input as result
	Mat result = frame.clone();

	// write modified data to result
	cudaMemcpy(result.data, device_output, frameDataSize, cudaMemcpyDeviceToHost);
	cudaCheckError();

	cudaFree(device_input);
	cudaFree(device_output);

	return result;
}

int main(int, char**)
{

	VideoCapture cap("H:\\Benutzer\\gpgpu\\OpenCVReadVideo\\Videos\\robotica_1080.mp4");

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

	for (;;)
	{	
		
		if (frame.dims == 0) { // we're done
			break;
		}

		output = modifyFrame(frame);

		// ------------------------------------------------
		//cvtColor(frame, edges, COLOR_BGR2GRAY);
		//GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
		//Sobel(frame, edges, frame.depth(), 2, 2);
		//Canny(edges, edges, 0, 30, 3);
		//imshow("edges", edges);
		// ------------------------------------------------

		// show the output from device
		imshow("edges", output);
		if (waitKey(1) >= 0) break;

		cap >> frame;
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	getchar();
	return 0;
}