#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <fstream>
#include <ctime>

#include "func.cuh"

const int layer = 4, RGB = 1;

int main() {
	int WeightSum = 0, NeuralSum = 0, n[layer] = { 1, 2, 4, 784 }, int dop = 0,
		DisWeightSum = 0, DisNeuralSum = 0, nc[layer] = { 784, 64, 16, 1 };
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

	float* outO = new float[n[layer - 1] * RGB];                //оптимальный вариант генератора в RAM
	float* DisoutO = new float[nc[layer - 1] * RGB];            //оптимальный вариант дискриминатора в RAM
	float* DisResult = new float[nc[layer - 1] * RGB];          //решение дискриминатора

	for (int i = 0; i < n[layer - 1] * RGB; i++)
		outO[i] = 0;
	for (int i = 0; i < nc[layer - 1] * RGB; i++)
		DisoutO[i] = 0;

	cudaMalloc((void**)&out, NeuralSum * RGB * sizeof(float));
	cudaMalloc((void**)&del, (NeuralSum - n[0]) * sizeof(float));
	cudaMalloc((void**)&weight, WeightSum * RGB * sizeof(float));
	cudaMalloc((void**)&delw, WeightSum * RGB * sizeof(float));
	cudaMalloc((void**)&Inp, n[0] * RGB * sizeof(float));                              //входные данные генератора в GPU RAM
	cudaMalloc((void**)&Oout, n[layer - 1] * RGB * sizeof(float));                     //оптимальный вариант генератора в GPU RAM

	cudaMalloc((void**)&Disout, DisNeuralSum * RGB * sizeof(float));
	cudaMalloc((void**)&Disdel, (DisNeuralSum - nc[0]) * sizeof(float));
	cudaMalloc((void**)&Disweight, DisWeightSum * RGB * sizeof(float));
	cudaMalloc((void**)&Disdelw, DisWeightSum * RGB * sizeof(float));
	cudaMalloc((void**)&DisInp, nc[0] * RGB * sizeof(float));                          //входные данные дискриминатора в GPU RAM
	cudaMalloc((void**)&DisOout, nc[layer - 1] * RGB * sizeof(float));                 //оптимальный вариант дискриминатора в GPU RAM

	DelwNull << < WeightSum, 1 >> > (delw, WeightSum);
	DelwNull << < DisWeightSum, 1 >> > (Disdelw, DisWeightSum);

	cv::Mat image = cv::imread("E:\\Foton\\ngnl_data\\training\\0\\1 (9).png");
	cv::Mat result(image.rows, image.cols, CV_8UC1);

	//InputOutImage(n[layer - 1], outO, Oout, image, RGB);
	DataCheck(WeightSum, weight, delw, 0, RGB);
	DataCheck(DisWeightSum, Disweight, Disdelw, 1, RGB);

	t1 = clock();
	for (int adm = 0; adm < 1000; adm++) {
		std::cout << "Iter #" << adm + 1 << std::endl;
		for (int l = 0; l < 1; l++) {
			for (int num = 0; num < 1; num++) {
				for (int k = 0; k < 5000; k++) {
					cv::Mat image = cv::imread("E:\\Foton\\ngnl_data\\training\\" + std::to_string(1) + "\\1 (" + std::to_string(k + 1) + ").png");
					OptDis(DisoutO, nc[layer - 1], RGB, DisOout, 1);

					InputInputImage(nc[0], Disout, DisoutO, DisInp, image, DisNeuralSum, RGB);
					GlobalSumFunc(nc, layer, DisNeuralSum, DisWeightSum, Disweight, Disout, RGB);
					ImageResult(DisNeuralSum, Disout, nc[layer - 1], RGB);
					GenIteration(nc, layer, DisNeuralSum, DisWeightSum, Disweight, Disout, Disdelw, DisOout, DisoutO, Disdel, RGB);

					Random(nc[0], DisNeuralSum, DisInp, Disout, RGB);
					OptDis(DisoutO, nc[layer - 1], RGB, DisOout, 0);
					GlobalSumFunc(nc, layer, DisNeuralSum, DisWeightSum, Disweight, Disout, RGB);
					DisIteration(nc, layer, DisNeuralSum, DisWeightSum, Disweight, Disout, Disdelw, DisOout, DisoutO, Disdel, RGB);
					ImageResult(DisNeuralSum, Disout, nc[layer - 1], RGB);
					//std::cout << "12";
					//ImageResult(DisNeuralSum, Disout, nc[layer - 1], RGB);
				}
				dop = 1;
				//std::cout << "12";

				for (int k = 0; k < 1; k++) {
					cv::Mat image = cv::imread("E:\\Foton\\ngnl_data\\training\\" + std::to_string(0) + "\\1 (" + std::to_string(k + 1) + ").png");
					/*Input(n[layer - 1], outO, Oout, image);*/
					// Random(n[0], NeuralSum, Inp, out);
					/*InputDiffer(dop, n, layer, WeightSum, nc[0], DisNeuralSum, NeuralSum, DisInp, Disout, weight, out, outO, dop, Inp);
					GlobalSumFunc(nc, layer, DisNeuralSum, DisWeightSum, Disweight, Disout);
					ImageOpt(DisNeuralSum, Disout, nc[layer - 1], outO);*/
					//InputOutImage(n[layer - 1], outO, Oout, image);
					//Random(n[0], NeuralSum, Inp, out, RGB);
					Random(n[0], NeuralSum, Inp, out, RGB);
					//GlobalSumFunc(n, layer, NeuralSum, WeightSum, weight, out, RGB);
					//InputData << <n[0], 1 >> > (Inp, out, n[0], 0, 0);
					InputOutImage(n[layer - 1], outO, Oout, image, RGB);
					GlobalSumFunc(n, layer, NeuralSum, WeightSum, weight, out, RGB);
					GenIteration(n, layer, NeuralSum, WeightSum, weight, out, delw, Oout, outO, del, RGB);
					Out(NeuralSum, layer, n, weight, out, result);
					//OutOutImage(NeuralSum, layer, n, out, result, RGB);
					//std::cout << WeightSum << std::endl;
					//cv::waitKey(500);
				}
			}
		}
		Backup(WeightSum, weight, delw, 0, RGB);
		Backup(DisWeightSum, Disweight, Disdelw, 1, RGB);
	}
}
