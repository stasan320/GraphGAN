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

const int coat = 4;

int main() {
	int WeightSum = 0, NeuralSum = 0, n[coat] = { 2, 3, 5, 1 }, Wnum = 0, Onum = 0, Dnum = 0;
	float* del, * delw = NULL, * weight, * out, * Inp, * Oout;
	int addvar, dop = 0;
	float pixel = 0;
	clock_t t1, t2;
	string name = "E:\\Foton\\ngnl_data\\training\\", filename;

	for (int i = 0; i < coat; i++)
		NeuralSum = n[i] + NeuralSum;
	std::cout << "Neurals: " << NeuralSum << std::endl;

	for (int i = 0; i < coat - 1; i++)
		WeightSum = n[i] * n[i + 1] + WeightSum;
	std::cout << "Weights: " << WeightSum << std::endl;
	
	for (int i = 0; i < coat - 1; i++)
		dop = n[i] + dop;

	float* weights = new float[WeightSum];
	float* InputDataArr = new float[n[0]];
	float* outO = new float[n[coat - 1]];
	/*for (int i = 0; i < n[0]; i++)
		InputDataArr[i] = 0;
	InputDataArr[0] = 0.7;*/
	//InputDataArr[1] = 0.3;
	/*InputDataArr[2] = 1;
	InputDataArr[3] = 0.563;*/

	/*for (int i = 0; i < n[coat - 1]; i++)
		outO[i] = 0;
	outO[0] = 0;*/
	//outO[1] = 1;

	cudaMalloc((void**)&out, NeuralSum * sizeof(float));
	cudaMalloc((void**)&del, (NeuralSum - n[0]) * sizeof(float));
	cudaMalloc((void**)&weight, WeightSum * sizeof(float));
	cudaMalloc((void**)&delw, WeightSum * sizeof(float));
	cudaMalloc((void**)&Inp, n[0] * sizeof(float));
	cudaMalloc((void**)&Oout, n[coat - 1] * sizeof(float));

	cudaMemcpy(Inp, InputDataArr, n[0] * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Oout, outO, n[coat - 1] * sizeof(float), cudaMemcpyHostToDevice);

	WeightCreation << <WeightSum, 1 >> > (weight, WeightSum);
	//InputData << <n[0], 1 >> > (Inp, out, n[0]);

	t1 = clock();
	for (int ad = 0; ad < 5000; ad++) {
		for (int num = 0; num < n[coat - 2]; num++) {
			ifstream nam(name + to_string(num) + ".txt");
			for (int k = 0; k < 2; k++) {
				/*nam >> filename;
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
				}*/

				/*cudaMemcpy(Inp, InputDataArr, n[0] * sizeof(float), cudaMemcpyHostToDevice);
				cudaMemcpy(Oout, outO, n[coat - 1] * sizeof(float), cudaMemcpyHostToDevice);*/
				//InputData << <n[0], 1 >> > (Inp, out, n[0]);

				InputDataArr[0] = k;
				InputDataArr[1] = 1 - k;
				cudaMemcpy(Inp, InputDataArr, n[0] * sizeof(float), cudaMemcpyHostToDevice);
				InputData << <n[0], 1 >> > (Inp, out, n[0]);
				outO[0] = 1 - k;
				cudaMemcpy(Oout, outO, n[coat - 1] * sizeof(float), cudaMemcpyHostToDevice);

				for (int i = 0; i < coat - 1; i++) {
					Sumfunc << <n[i + 1], 1 >> > (n[i], Wnum, Onum, weight, out, n[i + 1]);										//int coat, int Wnum, int Onum, float* weight, float* out
					Wnum = Wnum + n[i] * n[i + 1];
					Onum = Onum + n[i];
				}

				Onum = NeuralSum - n[coat - 1];
				Delta << <n[coat - 1], 100 >> > (Oout, out, del, Onum, n[coat - 1]);
				Wnum = WeightSum;

				for (int j = 0; j < coat - 2; j++) {
					//addvar = coat - 1 - j;
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
					//addvar = coat - 1 - j;
					Wnum = Wnum - n[coat - 1 - j] * n[coat - 2 - j];
					Deltaw << < n[coat - 2 - j], 1 >> > (weight, del, out, delw, Dnum, Onum, Wnum, n[coat - 1 - j], n[coat - 2 - j]);						//float* weight, float* del, float* out, float* delw, int Dnum, int Onum, int Wnum, int coat
					//std::cout << Wnum << endl;
					Dnum = Dnum + n[coat - 1 - j];
				}

				Wnum = 0;
				Onum = 0;
				Dnum = 0;
				//outO[num] = 0;

				cudaMemcpy(weights, out, (NeuralSum) * sizeof(float), cudaMemcpyDeviceToHost);
				for (int i = dop; i < NeuralSum; i++) cout << weights[i] << endl;
			}
			cout << endl;
		}
	}

	/*for (int num = 0; num < n[coat - 1]; num++) {
		Wnum = 0;
		Onum = 0;
		ifstream nam(name + to_string(num) + ".txt");
		nam >> filename;
		//std::cout << filename << endl;
		Mat image = imread(name + to_string(num) + "\\" + filename);
		/*cout << name + to_string(num) + "\\" + filename << endl;
		namedWindow("Display");
		imshow("Display", image);
		waitKey(1);
		Sleep(100);*/
		/*for (int i = 0; i < image.cols; i++) {
			for (int j = 0; j < image.rows; j++) {
				for (int p = 0; p < 3; p++) {
					pixel = pixel + image.at<Vec3b>(i, j)[p];
				}
				InputDataArr[i * image.rows + j] = pixel / 765;
				pixel = 0;
				//cout << InputDataArr[i * image.rows + j] << endl;
			}
		}

		cudaMemcpy(Inp, InputDataArr, n[0] * sizeof(float), cudaMemcpyHostToDevice);
		//cudaMemcpy(Oout, outO, n[coat - 1] * sizeof(float), cudaMemcpyHostToDevice);
		InputData << <n[0], 1 >> > (Inp, out, n[0]);

		for (int i = 0; i < coat - 1; i++) {
			Sumfunc << <n[i + 1], 1 >> > (n[i], Wnum, Onum, weight, out, n[i + 1]);										//int coat, int Wnum, int Onum, float* weight, float* out
			Wnum = Wnum + n[i] * n[i + 1];
			Onum = Onum + n[i];
		}

		cudaMemcpy(weights, out, (NeuralSum) * sizeof(float), cudaMemcpyDeviceToHost);
		for (int i = dop; i < NeuralSum; i++) cout << weights[i] << endl;
		cout << endl;
	}*/
	std::cout << "Time: " << clock() - t1 << std::endl;
	/*cudaMemcpy(weights, out, (NeuralSum) * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = dop; i < NeuralSum; i++) cout << weights[i] << endl;*/
	//Sleep(5000);
}
