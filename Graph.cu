#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <ctime>

#include "func.cuh"

const int layer = 3;

int main() {
	int WeightSum = 0, NeuralSum = 0, n[layer] = { 784, 15, 10 /*enter*/}, Wnum = 0, Onum = 0, Dnum = 0, dop = 0;
	float* del, * delw, * weight, * out, * Inp, * Oout, pixel = 0, max = 10, min = -10;
	clock_t t1;

	for (int i = 0; i < layer; i++)
		NeuralSum = n[i] + NeuralSum;
	std::cout << "Neurals: " << NeuralSum << std::endl;

	for (int i = 0; i < layer - 1; i++)
		WeightSum = n[i] * n[i + 1] + WeightSum;
	std::cout << "Weights: " << WeightSum << std::endl;

	float* weights = new float[NeuralSum];
	float* InputDataArr = new float[n[0]];
	float* outO = new float[n[layer - 1]];

	for (int i = 0; i < n[layer - 1]; i++)
		outO[i] = 0;

	cudaMalloc((void**)&out, NeuralSum * sizeof(float));
	cudaMalloc((void**)&del, (NeuralSum - n[0]) * sizeof(float));
	cudaMalloc((void**)&weight, WeightSum * sizeof(float));
	cudaMalloc((void**)&delw, WeightSum * sizeof(float));
	cudaMalloc((void**)&Inp, n[0] * sizeof(float));
	cudaMalloc((void**)&Oout, n[layer - 1] * sizeof(float));

	WeightGen << <WeightSum, 1 >> > (weight, WeightSum);
	DelwNull << < WeightSum, 1 >> > (delw, WeightSum);

	//cv::Mat image = cv::imread("E:\\1.jpg");
	cv::Mat result(1200, 1200, CV_8UC1);
	cv::Mat result(image.rows, image.cols, CV_8UC1);

	t1 = clock();
	for (int num = 0; num < 500; num++) {
		for (int k = 0; k < 10; k++) {
			cv::Mat image = cv::imread("E:\\Foton\\ngnl_data\\training\\" + std::to_string(k) + "\\1 (" + std::to_string(num) + ").png");
			OutputData(n, layer, outO, Oout, image, result, InputDataArr, Inp, out);
			NumberInp(out, image, InputDataArr, n, layer);
			Iteration(n, layer, NeuralSum, WeightSum, weight, out, delw, Oout, outO, del);
			Out(NeuralSum, layer, n, weights, out, image, result);
		}
	}

	std::cout << "Time " << clock() - t1 << std::endl;

	for (int i = 0; i < 100; i++) {
		//InputDataArr[0] = 0.524;
		std::cin >> InputDataArr[0];
		cudaMemcpy(Inp, InputDataArr, n[0] * sizeof(float), cudaMemcpyHostToDevice);
		InputData << <n[0], 1 >> > (Inp, out, n[0]);
		for (int i = 0; i < (layer - 1); i++) {
			Sumfunc << <n[i + 1], 1 >> > (n[i], Wnum, Onum, weight, out, n[i + 1]);										//int layer, int Wnum, int Onum, float* weight, float* out
			Wnum = Wnum + n[i] * n[i + 1];
			Onum = Onum + n[i];
		}

		Out(NeuralSum, layer, n, weights, out, result, result);
		Onum = 0;
		Wnum = 0;
	}
}
