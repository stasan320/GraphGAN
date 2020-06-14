#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;
using namespace cv;

__global__ void Inversion_CUDA(unsigned char* Image, int Channels) {
	int x = blockIdx.x;
	int y = blockIdx.y;
	int idx = (x + y * gridDim.x) * Channels;

	for (int i = 0; i < Channels; i++) {
		Image[idx + i] = 255 - Image[idx + i];
	}
}

int main() {
	Mat Input_Image = imread("E:\\Foton\\ngnl_data\\training\\0\\1.png");

	cout << "Height: " << Input_Image.rows << ", Width: " << Input_Image.cols << ", Channels: " << Input_Image.channels() << endl;

	unsigned char* Dev_Input_Image = NULL;

	cudaMalloc((void**)&Dev_Input_Image, Input_Image.rows * Input_Image.cols * Input_Image.channels());
	cudaMemcpy(Dev_Input_Image, Input_Image.data, Input_Image.rows * Input_Image.cols * Input_Image.channels(), cudaMemcpyHostToDevice);

	dim3 Grid_Image(Input_Image.rows * Input_Image.cols);
	Inversion_CUDA << <Grid_Image, 1 >> > (Dev_Input_Image, Input_Image.channels());
	cudaMemcpy(Input_Image.data, Dev_Input_Image, Input_Image.rows * Input_Image.cols * Input_Image.channels(), cudaMemcpyDeviceToHost);

	cudaFree(Dev_Input_Image);

	imwrite("E:\\image.png", Input_Image);
	return 0;
}
