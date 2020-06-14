#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;
using namespace cv;

__global__ void Inversion_CUDA(unsigned char* Image, int Channels, int* admin) {
	int pixel = 0;
	int x = blockIdx.x;
	int y = blockIdx.y;
	int idx = (x + y * gridDim.x) * Channels;

	for (int i = 0; i < Channels; i++) pixel = pixel + Image[idx + i];
	admin[x + y * gridDim.x] = pixel;
}

int main() {
	//Mat Input_Image = imread("E:\\Foton\\ngnl_data\\training\\0\\1.png");
	Mat Input_Image = imread("E:\\photo.png");
	unsigned int N = Input_Image.rows * Input_Image.cols;
	unsigned int K = N * 3;

	cout << "Height: " << N << ", Width: " << Input_Image.cols << ", Channels: " << Input_Image.channels() << endl;

	unsigned char* Dev_Input_Image = NULL;
	int l = 0;
	int pixels = 0;
	int* pixel;
	int* Image = new int[N];

	cudaMalloc((void**)&Dev_Input_Image, K);
	cudaMemcpy(Dev_Input_Image, Input_Image.data, K, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&pixel, N);

	dim3 Grid_Image(Input_Image.rows * Input_Image.cols);
	Inversion_CUDA << <Grid_Image, 1 >> > (Dev_Input_Image, Input_Image.channels(), pixel);
	/*for (int i = 0; i < Input_Image.rows; i++) {
		for (int k = 0; k < Input_Image.cols; k++) {
			for (int j = 0; j < 3; j++) pixels = pixels + Input_Image.at<Vec3b>(i, k)[j];
			Image[l] = pixels;
			l++;
			pixels = 0;
		}
	}*/

	//cudaMemcpy(Input_Image.data, Dev_Input_Image, Input_Image.rows * Input_Image.cols * Input_Image.channels(), cudaMemcpyDeviceToHost);
	cudaMemcpy(Image, pixel, N, cudaMemcpyDeviceToHost);

	cudaFree(Dev_Input_Image);
	cudaFree(pixel);

	//for (int i = 0; i < N; i++) cout << Image[i] << endl;

	//imwrite("E:\\image.png", Input_Image);
	return 0;
}
