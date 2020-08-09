//Create by Stasan

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <fstream>
#include <ctime>

#include "func.cuh"

const int layer = 4, RGB = 1;

int main() {
	int WeightSum = 0, NeuralSum = 0, n[layer] = { 4, 16, 4, 784 }, int dop = 0,
		DisWeightSum = 0, DisNeuralSum = 0, nc[layer] = { 784, 64, 16, 1 };
	float * del, * delw, * weight, * out, * Inp, * Oout, * DOout,
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
	cudaMalloc((void**)&DOout, RGB * sizeof(float));                                   //оптимальный вариант генератора в GPU RAM

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
				for (int k = 0; k < 10; k++) {
					//---------Input image data---//
					//---------SumFunc------------//
					//---------Out data-----------//
					//---------Opt out data-------//
					//---------Iter---------------//

					//---------Input result data--//
					//---------SumFunc------------//
					//---------Out data-----------//
					//---------Opt out data-------//
					//---------Iter---------------//

					cv::Mat image = cv::imread("E:\\Foton\\ngnl_data\\training\\" + std::to_string(1) + "\\1 (" + std::to_string(k + 1) + ").png");
					InputInputImage(nc[0], Disout, DisoutO, DisInp, image, DisNeuralSum, RGB);
					GlobalSumFunc(nc, layer, DisNeuralSum, DisWeightSum, Disweight, Disout, RGB);
					//ImageResult(DisNeuralSum, Disout, nc[layer - 1], RGB);
					OptDis(DisoutO, nc[layer - 1], RGB, DisOout, 1);
					DisIteration(nc, layer, DisNeuralSum, DisWeightSum, Disweight, Disout, Disdelw, DisOout, DisoutO, Disdel, RGB);

					//Random(nc[0], DisNeuralSum, DisInp, Disout, RGB);
					Random(n[0], NeuralSum, Inp, out, RGB);
					GlobalSumFunc(n, layer, NeuralSum, WeightSum, weight, out, RGB);
					Convert(DisInp, nc[0], NeuralSum, DisNeuralSum, out, RGB);
					for (int i = 0; i < RGB; i++)
						InputData << <nc[0], 1 >> > (DisInp, Disout, nc[0], i, DisNeuralSum);

					GlobalSumFunc(nc, layer, DisNeuralSum, DisWeightSum, Disweight, Disout, RGB);
					//ImageResult(DisNeuralSum, Disout, nc[layer - 1], RGB);
					OptDis(DisoutO, nc[layer - 1], RGB, DisOout, 0);
					DisIteration(nc, layer, DisNeuralSum, DisWeightSum, Disweight, Disout, Disdelw, DisOout, DisoutO, Disdel, RGB);
				}
				std::cout << "data";
				for (int k = 0; k < 10000; k++) {
					Random(n[0], NeuralSum, Inp, out, RGB);
					GlobalSumFunc(n, layer, NeuralSum, WeightSum, weight, out, RGB);
					Convert(DOout, nc[layer - 1], DisNeuralSum, 0, Disout, RGB);
					GenIteration(n, layer, NeuralSum, WeightSum, weight, out, delw, DOout, outO, del, RGB);
					Out(NeuralSum, layer, n, weight, out, result);
				}
			}
		}
		Backup(WeightSum, weight, delw, 0, RGB);
		Backup(DisWeightSum, Disweight, Disdelw, 1, RGB);
	}
}
