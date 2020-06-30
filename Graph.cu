#include <windows.h>
#include <iostream>
#include <cmath>
#include "func.cuh"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ctime>
#include <fstream>
#include <cstring>

using namespace cv;

const int layer = 3;

int main() {
	int WeightSum = 0, NeuralSum = 0, n[layer] = { 2, 2, 1 }, Wnum = 0, Onum = 0, Dnum = 0, dop = 0;
	float* del, * delw, * weight, * out, * Inp, * Oout, pixel = 0;
	clock_t t1;
	//std::string name = "E:\\Foton\\ngnl_data\\training\\", filename;

	for (int i = 0; i < layer; i++)
		NeuralSum = n[i] + NeuralSum;
	std::cout << "Neurals: " << NeuralSum << std::endl;

	for (int i = 0; i < layer - 1; i++)
		WeightSum = n[i] * n[i + 1] + WeightSum;
	std::cout << "Weights: " << WeightSum << std::endl;

	float* weights = new float[WeightSum];
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

	t1 = clock();
	for (int ad = 0; ad < 1; ad++) {
		for (int num = 0; num < 500; num++) {
			//std::ifstream nam(name + std::to_string(num) + ".txt");
			for (int k = 0; k < 2; k++) {
				InputDataArr[0] = 1 - k;
				InputDataArr[1] = k;
				cudaMemcpy(Inp, InputDataArr, n[0] * sizeof(float), cudaMemcpyHostToDevice);
				InputData << <n[0], 1 >> > (Inp, out, n[0]);
				outO[0] = 1 - k;
				cudaMemcpy(Oout, outO, n[layer - 1] * sizeof(float), cudaMemcpyHostToDevice);

				//Clayer << < 49, 1 >> > (weight, out, n[0]);
				for (int i = 0; i < (layer - 1); i++) {
					Sumfunc << <n[i + 1], 1 >> > (n[i], Wnum, Onum, weight, out, n[i + 1]);										//int layer, int Wnum, int Onum, float* weight, float* out
					Wnum = Wnum + n[i] * n[i + 1];
					Onum = Onum + n[i];
				}

				Onum = NeuralSum - n[layer - 1];
				Delta << <n[layer - 1], 1 >> > (Oout, out, del, Onum, n[layer - 1]);
				Wnum = WeightSum;

				for (int j = 0; j < layer - 1; j++) {
					Onum = Onum - n[layer - 2 - j];
					Wnum = Wnum - n[layer - 2 - j] * n[layer - 1 - j];
					DeltaN << <n[layer - 2 - j], 1 >> > (Dnum, Wnum, Onum, del, weight, out, n[layer - 1 - j], n[layer - 2 - j]);					    //int Dnum, int Wnum, int Onum, float* del, float* weight, float* out
					Dnum = Dnum + n[layer - 1 - j];
				}

				Wnum = WeightSum;
				Dnum = 0;
				Onum = NeuralSum - n[layer - 1];

				for (int j = 0; j < layer - 1; j++) {
					Onum = Onum - n[layer - 2 - j];
					Wnum = Wnum - n[layer - 1 - j] * n[layer - 2 - j];
					Deltaw << < n[layer - 2 - j], 1 >> > (weight, del, out, delw, Dnum, Onum, Wnum, n[layer - 1 - j], n[layer - 2 - j]);				//float* weight, float* del, float* out, float* delw, int Dnum, int Onum, int Wnum, int layer, int n
					Dnum = Dnum + n[layer - 1 - j];
				}

				/*Dnum = NeuralSum - n[0] - n[1];
				ConvDeltaW << < n[1] / 16, 1 >> > (weight, out, del, delw, Dnum, n[1] / 49);*/

				Wnum = 0;
				Onum = 0;
				Dnum = 0;

				cudaMemcpy(weights, out, (NeuralSum) * sizeof(float), cudaMemcpyDeviceToHost);
				for (int i = (NeuralSum - n[layer - 1]); i < NeuralSum; i++) {
					//std::cout << InputDataArr[0] << "" << InputDataArr[1] << " ";
					std::cout << weights[i] << " " << outO[i] << std::endl;;
				}
			}
			std::cout << std::endl;
		}

	}
	std::cout << "Time: " << clock() - t1 << std::endl;
}
