#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <fstream>
#include <ctime>

#include "func.cuh"

const int layer = 5;

int main() {
	int WeightSum = 0, NeuralSum = 0, n[layer] = { 2, 32, 128, 512, 784 }, Wnum = 0, Onum = 0, Dnum = 0, dop;
	float* del, * delw, * weight, * Bweight, * out, * Inp, * Oout;
	clock_t t1;
	std::string filename;

	for (int i = 0; i < layer; i++)
		NeuralSum = n[i] + NeuralSum;
	std::cout << "Neurals: " << NeuralSum << std::endl;

	for (int i = 0; i < layer - 1; i++)
		WeightSum = n[i] * n[i + 1] + WeightSum;
	std::cout << "Weights: " << WeightSum << std::endl;
	std::cout << std::endl;

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

	DelwNull << < WeightSum, 1 >> > (delw, WeightSum);

	cv::Mat image = cv::imread("E:\\Foton\\ngnl_data\\training\\0\\1 (1).png");
	cv::Mat result(image.rows, image.cols, CV_8UC1);
	Input(n, layer, outO, Oout, image);

	DataCheck(WeightSum, weight, delw);

	t1 = clock();
	for (int adm = 0; adm < 100; adm++) {
		std::cout << "Iter #" << adm + 1 << std::endl;
		for (int l = 0; l < 5000; l++) {
			for (int num = 0; num < 1; num++) {
				for (int k = 0; k < 2; k++) {
					cv::Mat image = cv::imread("E:\\Foton\\ngnl_data\\training\\" + std::to_string(k) + "\\1 (" + std::to_string(num + 1) + ").png");
					/*cv::imshow("Out1", image);
					cv::waitKey(1);*/
					InputDataArr[k] = 1;
					cudaMemcpy(Inp, InputData, n[0] * sizeof(float), cudaMemcpyHostToDevice);
					InputData << <n[0], 1 >> > (Inp, out, n[0]);
					
					Input(n, layer, outO, Oout, image);
					Iteration(n, layer, NeuralSum, WeightSum, weight, out, delw, Oout, outO, del);
					//Out(NeuralSum, layer, n, weights, out, result);
					InputDataArr[k] = 0;
				}
			}
		}
		float* Bweight = new float[WeightSum];
		float* Bdelw = new float[WeightSum];
		cudaMemcpy(Bweight, weight, WeightSum * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(Bdelw, delw, WeightSum * sizeof(float), cudaMemcpyDeviceToHost);
		std::ofstream fweight("E:\\Foton\\ngnl_data\\backup\\weight.dat");
		std::ofstream fdelw("E:\\Foton\\ngnl_data\\backup\\delw.dat");

		for (int i = 0; i < WeightSum; i++) {
			fweight << Bweight[i] << " ";
			fdelw << Bdelw[i] << " ";
			//fout << i << " ";
		}
		std::cout << "Backup" << std::endl;
		delete[] Bweight;
		delete[] Bdelw;

		std::ofstream config("E:\\Foton\\ngnl_data\\backup\\config.txt");
		config << 1;
		config.close();
	}
}
