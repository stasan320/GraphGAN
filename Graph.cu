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

using namespace std;
using namespace cv;

const int coat = 3;

int main() {
	int WeightSum = 0, NeuralSum = 0, n[coat] = { 784, 400, 10 }, Wnum = 0, Onum = 0, Dnum = 0, dop = 0;
	float* del, * delw, * weight, * out, * Inp, * Oout;
	float pixel = 0;
	clock_t t1, t2;
	string name = "E:\\Foton\\ngnl_data\\training\\", filename;

	for (int i = 0; i < coat; i++)
		NeuralSum = n[i] + NeuralSum;
	std::cout << "Neurals: " << NeuralSum << std::endl;

	for (int i = 0; i < coat - 1; i++)
		WeightSum = n[i] * n[i + 1] + WeightSum;
	std::cout << "Weights: " << WeightSum << std::endl;

	float* weights = new float[WeightSum];
	float* InputDataArr = new float[n[0]];
	float* outO = new float[n[coat - 1]];
	
	for (int i = 0; i < n[coat - 1]; i++)
		outO[i] = 0;

	cudaMalloc((void**)&out, NeuralSum * sizeof(float));
	cudaMalloc((void**)&del, (NeuralSum - n[0]) * sizeof(float));
	cudaMalloc((void**)&weight, WeightSum * sizeof(float));
	cudaMalloc((void**)&delw, WeightSum * sizeof(float));
	cudaMalloc((void**)&Inp, n[0] * sizeof(float));
	cudaMalloc((void**)&Oout, n[coat - 1] * sizeof(float));

	WeightCreation << <WeightSum, 1 >> > (weight, WeightSum);
	DelwNull << < WeightSum, 1 >> > (delw, WeightSum);

	t1 = clock();
	for (int ad = 0; ad < 10; ad++) {
		for (int num = 0; num < 10; num++) {
			ifstream nam(name + to_string(num) + ".txt");
			for (int k = 0; k < 2; k++) {
				nam >> filename;
				outO[num] = 1;
				Mat image = imread(name + to_string(num) + "\\" + filename);
				for (int i = 0; i < image.cols; i++) {
					for (int j = 0; j < image.rows; j++) {
						for (int p = 0; p < 3; p++) {
							pixel = pixel + image.at<Vec3b>(i, j)[p];
						}
						InputDataArr[i * image.rows + j] = pixel / 765;
						pixel = 0;
					}
				}

				cudaMemcpy(Inp, InputDataArr, n[0] * sizeof(float), cudaMemcpyHostToDevice);
				cudaMemcpy(Oout, outO, n[coat - 1] * sizeof(float), cudaMemcpyHostToDevice);
				InputData << <n[0], 1 >> > (Inp, out, n[0]);

				for (int i = 0; i < (coat - 1); i++) {
					Sumfunc << <n[i + 1], 1 >> > (n[i], Wnum, Onum, weight, out, n[i + 1]);										//int coat, int Wnum, int Onum, float* weight, float* out
					Wnum = Wnum + n[i] * n[i + 1];
					Onum = Onum + n[i];
				}

				Onum = NeuralSum - n[coat - 1];
				Delta << <n[coat - 1], 1 >> > (Oout, out, del, Onum, n[coat - 1]);
				Wnum = WeightSum;

				for (int j = 0; j < coat - 1; j++) {
					Onum = Onum - n[coat - 2 - j];
					Wnum = Wnum - n[coat - 2 - j] * n[coat - 1 - j];
					DeltaN << <n[coat - 2 - j], 1 >> > (Dnum, Wnum, Onum, del, weight, out, n[coat - 1 - j], n[coat - 2 - j]);					    //int Dnum, int Wnum, int Onum, float* del, float* weight, float* out
					Dnum = Dnum + n[coat - 1 - j];
				}

				Wnum = WeightSum;
				Dnum = 0;
				Onum = NeuralSum - n[coat - 1];

				for (int j = 0; j < coat - 1; j++) {
					Onum = Onum - n[coat - 2 - j];
					Wnum = Wnum - n[coat - 1 - j] * n[coat - 2 - j];
					Deltaw << < n[coat - 2 - j], 1 >> > (weight, del, out, delw, Dnum, Onum, Wnum, n[coat - 1 - j], n[coat - 2 - j]);				//float* weight, float* del, float* out, float* delw, int Dnum, int Onum, int Wnum, int coat, int n
					Dnum = Dnum + n[coat - 1 - j];
				}

				Wnum = 0;
				Onum = 0;
				Dnum = 0;
				outO[num] = 0;

				cudaMemcpy(weights, out, (NeuralSum) * sizeof(float), cudaMemcpyDeviceToHost);
				for (int i = (NeuralSum - n[coat - 1]); i < NeuralSum; i++) {
					//std::cout << InputDataArr[0] << "" << InputDataArr[1] << " ";
					cout << weights[i] /*<< " " << outO[0]*/ << endl;;
				}
			}
			cout << endl;
		}
	}
	std::cout << "Time: " << clock() - t1 << std::endl;
}
