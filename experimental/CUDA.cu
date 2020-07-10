#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <fstream>
#include <ctime>

#include "func.cuh"

const int layer = 3;

int main() {
	int WeightSum = 0, NeuralSum = 0, n[layer] = { 16, 64, 784 }, int dop = 0, RGB = 3,
		DisWeightSum = 0, DisNeuralSum = 0, nc[layer] = { 784, 64, 1 };
	float * del, * delw, * weight, * out, * Inp, * Oout,
		  * Disdel, * Disdelw, * Disweight, * Disout, * DisInp, * DisOout;
	clock_t t1;
	std::string filename;

	/*---------перевести в const---------*/
	/*определение основных переменных Gen*/

	for (int i = 0; i < layer; i++)
		NeuralSum = n[i] + NeuralSum;
	std::cout << "Neurals: " << NeuralSum * RGB << std::endl;

	for (int i = 0; i < layer - 1; i++)
		WeightSum = n[i] * n[i + 1] + WeightSum;
	std::cout << "Weights: " << WeightSum * RGB << std::endl;

	for (int i = 0; i < layer; i++)
		DisNeuralSum = nc[i] + DisNeuralSum;
	std::cout << "Neurals: " << DisNeuralSum * RGB << std::endl;

	for (int i = 0; i < layer - 1; i++)
		DisWeightSum = nc[i] * nc[i + 1] + DisWeightSum;
	std::cout << "Weights: " << DisWeightSum * RGB << std::endl;
	std::cout << std::endl;

	/*---------перевести в const---------*/

	float* outO = new float[n[layer - 1] * RGB];
	float* DisoutO = new float[nc[layer - 1] * RGB];

	for (int i = 0; i < n[layer - 1] * RGB; i++)
		outO[i] = 0;
	for (int i = 0; i < nc[layer - 1] * RGB; i++)
		DisoutO[i] = 0;

	cudaMalloc((void**)&out, NeuralSum * RGB * sizeof(float));
	cudaMalloc((void**)&del, (NeuralSum - n[0]) * sizeof(float));
	cudaMalloc((void**)&weight, WeightSum * RGB * sizeof(float));
	cudaMalloc((void**)&delw, WeightSum * RGB * sizeof(float));
	cudaMalloc((void**)&Inp, n[0] * RGB * sizeof(float));
	cudaMalloc((void**)&Oout, n[layer - 1] * 1 * sizeof(float));

	cudaMalloc((void**)&Disout, DisNeuralSum * RGB * sizeof(float));
	cudaMalloc((void**)&Disdel, (DisNeuralSum - nc[0]) * sizeof(float));
	cudaMalloc((void**)&Disweight, DisWeightSum * RGB * sizeof(float));
	cudaMalloc((void**)&Disdelw, DisWeightSum * RGB * sizeof(float));
	cudaMalloc((void**)&DisInp, nc[0] * RGB * sizeof(float));
	cudaMalloc((void**)&DisOout, nc[layer - 1] * RGB * sizeof(float));

	DelwNull << < WeightSum, 1 >> > (delw, WeightSum);
	DelwNull << < DisWeightSum, 1 >> > (Disdelw, DisWeightSum);

	cv::Mat image = cv::imread("E:\\Foton\\ngnl_data\\training\\0\\1 (9).png");
	cv::Mat result(image.rows, image.cols, CV_8UC1);
	InputOutImage(n[layer - 1], outO, Oout, image);
	DataCheck(WeightSum, weight, delw, 0);
	DataCheck(DisWeightSum, Disweight, Disdelw, 1);

	t1 = clock();
	for (int adm = 0; adm < 1000; adm++) {
		std::cout << "Iter #" << adm + 1 << std::endl;
		for (int l = 0; l < 1; l++) {
			for (int num = 0; num < 1; num++) {
				/*for (int k = 0; k < 5000; k++) {
					/*---------Gen from Iter---------*/
					/*InputDiffer(dop, n, layer, WeightSum, nc[0], DisNeuralSum, NeuralSum, DisInp, Disout, weight, out, outO, dop, Inp);
					GlobalSumFunc(nc, layer, DisNeuralSum, DisWeightSum, Disweight, Disout);
					for (int i = 0; i < RGB; i++)
						DisoutO[i] = 0;
					DisIteration(nc, layer, DisNeuralSum, DisWeightSum, Disweight, Disout, Disdelw, DisOout, DisoutO, Disdel, DisInp);
					/*ImageResult(DisNeuralSum, Disout, nc[layer - 1]);
					std::cout << std::endl;*/

					/*---------Opt from Iter---------*/
					/*cv::Mat image = cv::imread("E:\\Foton\\ngnl_data\\training\\" + std::to_string(0) + "\\1 (" + std::to_string(k + 1) + ").png");
					InputInputImage(nc[0], Disout, outO, DisInp, image, DisNeuralSum);
					GlobalSumFunc(nc, layer, DisNeuralSum, DisWeightSum, Disweight, Disout);
					for (int i = 0; i < RGB; i++)
						DisoutO[i] = 1;
					DisIteration(nc, layer, DisNeuralSum, DisWeightSum, Disweight, Disout, Disdelw, DisOout, DisoutO, Disdel, DisInp);
					/*ImageResult(DisNeuralSum, Disout, nc[layer - 1]);
					std::cout << std::endl;*/
				/*}
				dop = 1;*/

				for (int k = 0; k < 5000; k++) {
					/*cv::Mat image = cv::imread("E:\\Foton\\ngnl_data\\training\\" + std::to_string(k) + "\\1 (" + std::to_string(num + 1) + ").png");
					Input(n[layer - 1], outO, Oout, image);*/
					// Random(n[0], NeuralSum, Inp, out);
					/*InputDiffer(dop, n, layer, WeightSum, nc[0], DisNeuralSum, NeuralSum, DisInp, Disout, weight, out, outO, dop, Inp);
					GlobalSumFunc(nc, layer, DisNeuralSum, DisWeightSum, Disweight, Disout);
					ImageOpt(DisNeuralSum, Disout, nc[layer - 1], outO);*/
					//InputOutImage(n[layer - 1], outO, Oout, image);
					Random(n[0], NeuralSum, Inp, out);
					GenIteration(n, layer, NeuralSum, WeightSum, weight, out, delw, Oout, outO, del);
					OutOutImage(NeuralSum, layer, n, out, result);
					//std::cout << WeightSum << std::endl;
				}
			}
		}
		Backup(WeightSum, weight, delw, 0);
		Backup(DisWeightSum, Disweight, Disdelw, 1);
	}
}
