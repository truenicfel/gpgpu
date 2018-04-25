// OpenCVReadVideo.cpp : Definiert den Einstiegspunkt für die Konsolenanwendung.
//

#include <iostream>
#include "stdafx.h"

#include "opencv2/opencv.hpp"
#include "cuda_runtime.h"

using namespace cv;
using namespace std;

extern void addKernel(int *c, const int *a, const int *b);


void doCuda()
{
	const int a[5] = { 1, 2, 3, 4, 5 };
	const int b[5] = { 10, 20, 30, 40, 50 };
	int c[5] = { 0 };
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;

	cudaMalloc((void**)&dev_c, 5 * sizeof(int));
	cudaMalloc((void**)&dev_a, 5 * sizeof(int));
	cudaMalloc((void**)&dev_b, 5 * sizeof(int));

	cudaMemcpy(dev_a, a, 5 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, 5 * sizeof(int), cudaMemcpyHostToDevice);

	void* args1[] = { &dev_c, &dev_a, &dev_b };
	cudaLaunchKernel<void>(&addKernel, dim3(1), dim3(5), args1);

	cudaDeviceSynchronize();
	cudaMemcpy(c, dev_c, 5 * sizeof(int), cudaMemcpyDeviceToHost);

	printf("%i  %i  %i  %i  %i", c[0], c[1], c[2], c[3], c[4]);

	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
}

int main(int, char**)
{
	doCuda();


//	VideoCapture cap("Z:/Videos/robotica_1080.mp4"); // open the default camera
	VideoCapture cap("H:\\Benutzer\\Dokumente\\OpenCVReadVideo\\Videos\\robotica_1080.mp4");
//	VideoCapture cap("C:/Users/fischer/Downloads/Bennu4k169Letterbox_h264.avi"); // open the default camera
//	VideoCapture cap("D:/Users/fischer/Videos/fireworks.mp4");
//	VideoCapture cap("D:/Users/fischer/Videos/Bennu4k169Letterbox_h264.mp4");
//	VideoCapture cap("D:/Users/fischer/Videos/Bennu4k169Letterbox_h264.avi");

	if (!cap.isOpened())  // check if we succeeded
		return -1;

	Mat edges;
	namedWindow("edges", 1);
	bool firstCall = true;
	for (;;)
	{	
		Mat frame;
		Mat output;
		cap >> frame; // get a new frame from camera
		if (frame.dims == 0) { // we're done
			break;
		}

		if (firstCall)
		{
			cout << "frame: dims: " << frame.dims << ", size[0]: " << frame.size[0] << ", size[1]:" << frame.size[1] << ", step[0]: " << frame.step[0] << ", step[1]:" << frame.step[1];
			cout << ", type: " << frame.type() << " (CV16U: " << CV_16UC1 << ", CV8UC3: " << CV_8UC3 << ")" << ", elemSize: " << frame.elemSize();
			cout << ", rows: " << frame.rows << ", cols: " << frame.cols << ", size: " << frame.size << ", dataPtr: " << frame.data << endl;
			firstCall = false;
		}

		output = frame.clone();

		//cvtColor(frame, edges, COLOR_BGR2GRAY);
		//GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
//		Sobel(frame, edges, frame.depth(), 2, 2);
//		Canny(edges, edges, 0, 30, 3);
//		imshow("edges", edges);
		imshow("edges", output);
		if (waitKey(1) >= 0) break;
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	getchar();
	return 0;
}