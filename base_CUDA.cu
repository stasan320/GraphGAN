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

void Image_Inversion_CUDA(unsigned char* Input_Image, int Height, int Width, int Channels) {
	unsigned char* Dev_Input_Image = NULL;

	cudaMalloc((void**)&Dev_Input_Image, Height * Width * Channels);
	cudaMemcpy(Dev_Input_Image, Input_Image, Height * Width * Channels, cudaMemcpyHostToDevice);

	dim3 Grid_Image(Width, Height);
	Inversion_CUDA << <Grid_Image, 1 >> > (Dev_Input_Image, Channels);
	cudaMemcpy(Input_Image, Dev_Input_Image, Height * Width * Channels, cudaMemcpyDeviceToHost);

	cudaFree(Dev_Input_Image);
}

int main() {
	Mat Input_Image = imread("image.png");

	cout << "Height: " << Input_Image.rows << ", Width: " << Input_Image.cols << ", Channels: " << Input_Image.channels() << endl;
	system("pause");

	Image_Inversion_CUDA(Input_Image.data, Input_Image.rows, Input_Image.cols, Input_Image.channels());

	imwrite("image.png", Input_Image);
	return 0;
}
