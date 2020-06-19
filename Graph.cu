#include <windows.h>
#include <iostream>
#include <cmath>
#include "func.cuh"

using namespace std;

const int coat = 4;

int main() {
	int WeightSum = 0, NeuralSum = 0, n[coat] = { 3, 4, 2, 3 }, Wnum = 0, Onum = 0;
	float* del = NULL, * delw = NULL, * weight, * out, * Inp, * Oout;
	int addvar;

	for (int i = 0; i < coat; i++)   
		NeuralSum = n[i] + NeuralSum;

	for (int i = 0; i < coat - 1; i++) 
		WeightSum = n[i] * n[i + 1] + WeightSum;

	float* weights = new float[NeuralSum];
	float* InputDataArr = new float[n[0]];
	float* outO = new float[n[coat - 1]];
	InputDataArr[0] = 0.7;
	InputDataArr[1] = 0.3;
	InputDataArr[2] = 1;

	outO[0] = 0.4;
	outO[1] = 1;
	outO[2] = 0.6;
	addvar = n[coat - 1];
	/*outO[3] = 0.785;
	outO[4] = 1;*/

	cudaMalloc((void**)&out, NeuralSum * sizeof(float));
	cudaMalloc((void**)&del, (NeuralSum - n[0]) * sizeof(float));
	cudaMalloc((void**)&weight, WeightSum * sizeof(float));
	cudaMalloc((void**)&delw, WeightSum * sizeof(float));
	cudaMalloc((void**)&Inp, n[0] * sizeof(float));
	cudaMalloc((void**)&Oout, addvar * sizeof(float));

	cudaMemcpy(Inp, InputDataArr, n[0] * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Oout, outO, n[coat - 1] * sizeof(float), cudaMemcpyHostToDevice);

	WeightCreation << <WeightSum, 1 >> > (weight, WeightSum);
	InputData << <n[0], 1 >> > (Inp, out, n[0]);
 
	cudaMemcpy(weights, out, NeuralSum * sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(out, InputData, NeuralS * sizeof(float), cudaMemcpyHostToDevice);
	for (int i = 0; i < n[0]; i++) cout << weights[i] << endl;

	cout << "Num #1" << endl;

	for (int i = 0; i < coat - 1; i++) {
		Sumfunc << <n[i], 1 >> > (n[i], Wnum, Onum, weight, out, n[i + 1]);										//int coat, int Wnum, int Onum, float* weight, float* out
		Wnum = Wnum + n[i] * n[i + 1];
		Onum = Onum + n[i];
	} 
	cudaMemcpy(weights, out, NeuralSum * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < n[coat - 1]; i++) cout << weights[Onum + i] << endl;

	addvar = n[coat - 1];
	Delta << <n[addvar], 1 >> > (weight, Oout, out, del, Onum, n[addvar]);									    //float* weight, float* outO, float* out, float* del, int Onum, int size

	/*cudaMemcpy(weights, del, n[coat - 1] * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < n[coat - 1]; i++) cout << weights[i] << endl;*/

	for (int i = coat - 1; i > 0; i--) {
		for (int j = 0; j < n[i]; j++) {
			Deltaw << <n[coat - i], 1 >> > (n[coat - i], delw, n[coat - i], del, out, Wnum, Onum, weight);			//int size, float* delw, int coat, float* del, float* out, int Wnum, int Onum, float* weight
			Wnum = Wnum + n[i];
		}
		Wnum = Wnum - n[i] * n[i - 1];
		Onum = Onum - n[i - 1];
	}
  }
