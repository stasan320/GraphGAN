#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <fstream>
#include <ctime>

#include "func.cuh"

const int layer = 4;

int main() {
	int WeightSum = 0, NeuralSum = 0, DisWeightSum = 0, DisNeuralSum = 0, n[layer] = { 16, 64, 128, 784 }, nc[layer] = {784, 512, 64, 1};
	float* del, * delw, * weight, * out, * Inp, * Oout, * Disweight, * Disout, * Disdelw, * DisOout, * Disdel, * DisInp, * Dis;
	clock_t t1;
	std::string filename;

	for (int i = 0; i < layer; i++)
		NeuralSum = n[i] + NeuralSum;
	std::cout << "Neurals: " << NeuralSum * 3 << std::endl;

	for (int i = 0; i < layer - 1; i++)
		WeightSum = n[i] * n[i + 1] + WeightSum;
	std::cout << "Weights: " << WeightSum * 3 << std::endl;

	for (int i = 0; i < layer; i++)
		DisNeuralSum = nc[i] + DisNeuralSum;
	std::cout << "Neurals: " << DisNeuralSum * 3 << std::endl;

	for (int i = 0; i < layer - 1; i++)
		DisWeightSum = nc[i] * nc[i + 1] + DisWeightSum;
	std::cout << "Weights: " << DisWeightSum * 3 << std::endl;
	std::cout << std::endl;

	float* outO = new float[n[layer - 1] * 3];
	float* DisoutO = new float[nc[layer - 1] * 3];

	/*for (int i = 0; i < n[layer - 1] * 3; i++)
		outO[i] = 0;*/

	cudaMalloc((void**)&out, NeuralSum * 3 * sizeof(float));
	cudaMalloc((void**)&del, (NeuralSum - n[0]) * sizeof(float));
	cudaMalloc((void**)&weight, WeightSum * 3 * sizeof(float));
	cudaMalloc((void**)&delw, WeightSum * 3 * sizeof(float));
	cudaMalloc((void**)&Inp, n[0] * 3 * sizeof(float));
	cudaMalloc((void**)&Oout, n[layer - 1] * 3 * sizeof(float));

	cudaMalloc((void**)&Disout, DisNeuralSum * 3 * sizeof(float));
	cudaMalloc((void**)&Disdel, (DisNeuralSum - nc[0]) * sizeof(float));
	cudaMalloc((void**)&Disweight, DisWeightSum * 3 * sizeof(float));
	cudaMalloc((void**)&Disdelw, DisWeightSum * 3 * sizeof(float));
	cudaMalloc((void**)&DisInp, nc[0] * 3 * sizeof(float));
	cudaMalloc((void**)&DisOout, nc[layer - 1] * 3 * sizeof(float));

	cudaMalloc((void**)&Dis, nc[layer - 1] * 3 * sizeof(float));

	DelwNull << < WeightSum, 1 >> > (delw, WeightSum, 0);
	DelwNull << < DisWeightSum, 1 >> > (Disdelw, DisWeightSum, 0);

	cv::Mat image = cv::imread("E:\\Foton\\ngnl_data\\training\\0\\1 (9).png");
	/*cv::imshow("Out", image);
	cv::waitKey(10000);*/
	cv::Mat result(image.rows, image.cols, CV_8UC3);
	Input(n[layer - 1], outO, Oout, image);
	DataCheck(WeightSum, weight, delw, 0);
	DataCheck(DisWeightSum, Disweight, Disdelw, 1);

	DelwNull << < nc[layer - 1], 1 >> > (Dis, nc[layer - 1], 1);

	t1 = clock();
	for (int adm = 0; adm < 1000; adm++) {
		std::cout << "Iter #" << adm + 1 << std::endl;
		for (int l = 0; l < 1; l++) {
			for (int num = 0; num < 5000; num++) {
				for (int k = 0; k < 1; k++) {
					InputGen(n[0], NeuralSum, Inp, out);
					SumGen(WeightSum, NeuralSum, DisNeuralSum, n, weight, out, result, Inp, Disout, layer);
					SumDis(DisWeightSum, DisNeuralSum, NeuralSum, nc, Disweight, Disout, Dis, image, Inp, layer);
					//cv::Mat image = cv::imread("E:\\Foton\\ngnl_data\\training\\" + std::to_string(k) + "\\1 (" + std::to_string(num + 1) + ").png");
					//Input(n[layer - 1], outO, Oout, image);
					IterationGen(n, layer, NeuralSum, WeightSum, weight, out, delw, Dis, del);
					Out(NeuralSum, layer, n, out, result);
				}
			}
		}
		Backup(WeightSum, weight, delw, 0);

		for (int l = 0; l < 1; l++) {
			for (int num = 0; num < 5000; num++) {
				InputGen(n[0], NeuralSum, Inp, out);
				SumGen(WeightSum, NeuralSum, DisNeuralSum, n, weight, out, result, Inp, Disout, layer);

				//InputGen(nc[0], DisNeuralSum, DisInp, Disout);
				for (int i = 0; i < 3; i++)
					DisoutO[i] = 0.3;
				cudaMemcpy(DisOout, DisoutO, nc[layer - 1] * 3 * sizeof(float), cudaMemcpyHostToDevice);
				IterationGen(nc, layer, DisNeuralSum, DisWeightSum, Disweight, Disout, Disdelw, DisOout, Disdel);
				/*ImageResult(DisNeuralSum, Disout, nc[layer - 1]);
				std::cout << std::endl;*/

				for (int i = 0; i < 3; i++)
					DisoutO[i] = 1;
				cudaMemcpy(DisOout, DisoutO, nc[layer - 1] * 3 * sizeof(float), cudaMemcpyHostToDevice);
				cv::Mat image = cv::imread("E:\\Foton\\ngnl_data\\training\\" + std::to_string(0) + "\\1 (" + std::to_string(num + 1) + ").png");
				/*cv::imshow("Out", image);
				cv::waitKey(1);*/
				InputImage(nc[0], Disout, DisInp, image, DisNeuralSum);
				IterationGen(nc, layer, DisNeuralSum, DisWeightSum, Disweight, Disout, Disdelw, DisOout, Disdel);
				/*ImageResult(DisNeuralSum, Disout, nc[layer - 1]);
				std::cout << std::endl;*/
			}
		}
		Backup(DisWeightSum, Disweight, Disdelw, 1);
	}
}
