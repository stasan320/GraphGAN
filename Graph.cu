#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <fstream>
#include <ctime>

#include "func.cuh"

const int layer = 3;

int main() {
	int WeightSum = 0, NeuralSum = 0, n[layer] = { 16, 64, 784 };
	float* del, * delw, * weight, * out, * Inp, * Oout;
	clock_t t1;
	std::string filename;

	/*---------перевести в const---------*/
	/*определение основных переменных Gen*/

	for (int i = 0; i < layer; i++)
		NeuralSum = n[i] + NeuralSum;
	std::cout << "Neurals: " << NeuralSum * 3 << std::endl;

	for (int i = 0; i < layer - 1; i++)
		WeightSum = n[i] * n[i + 1] + WeightSum;
	std::cout << "Weights: " << WeightSum * 3 << std::endl;
	std::cout << std::endl;

	/*---------перевести в const---------*/

	float* outO = new float[n[layer - 1] * 3];

	for (int i = 0; i < n[layer - 1] * 3; i++)
		outO[i] = 0;

	cudaMalloc((void**)&out, NeuralSum * 3 * sizeof(float));
	cudaMalloc((void**)&del, (NeuralSum - n[0]) * sizeof(float));
	cudaMalloc((void**)&weight, WeightSum * 3 * sizeof(float));
	cudaMalloc((void**)&delw, WeightSum * 3 * sizeof(float));
	cudaMalloc((void**)&Inp, n[0] * 3 * sizeof(float));
	cudaMalloc((void**)&Oout, n[layer - 1] * 3 * sizeof(float));

	DelwNull << < WeightSum, 1 >> > (delw, WeightSum);

	cv::Mat image = cv::imread("E:\\Foton\\ngnl_data\\training\\0\\1 (9).png");
	cv::Mat result(image.rows, image.cols, CV_8UC3);
	InputOutImage(n[layer - 1], outO, Oout, image);
	DataCheck(WeightSum, weight, delw, 0);

	t1 = clock();
	for (int adm = 0; adm < 1000; adm++) {
		std::cout << "Iter #" << adm + 1 << std::endl;
		for (int l = 0; l < 1; l++) {
			for (int num = 0; num < 5000; num++) {
				for (int k = 0; k < 1; k++) {
					/*cv::Mat image = cv::imread("E:\\Foton\\ngnl_data\\training\\" + std::to_string(k) + "\\1 (" + std::to_string(num + 1) + ").png");
					Input(n[layer - 1], outO, Oout, image);*/
					Random(n[0], NeuralSum, Inp, out, 0);
					GlobalSumFunc(n, layer, NeuralSum, WeightSum, weight, out);
					Iteration(n, layer, NeuralSum, WeightSum, weight, out, delw, Oout, outO, del);
					OutOutImage(NeuralSum, layer, n, out, result);
				}
			}
		}
		Backup(WeightSum, weight, delw, 0);
	}
}
