#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void WeightCreation(float* weight, int size) {                                                          //инициализация весов
	int index = blockIdx.x + blockIdx.y * gridDim.x;
	if (index < size)
		weight[index] = /*1 / (1 + exp2f(-index))*/0.384;
}

__global__ void InputData(float* data, float* out, int size) {
	int index = blockIdx.x + blockIdx.y * gridDim.x;
	if (index < size)
		out[index] = data[index];
}

__global__ void Sumfunc(int coat, int Wnum, int Onum, float* weight, float* out, int dop) {
	int index = blockIdx.x + blockIdx.y * gridDim.x;
	float net = 0;

	if (index < dop) {
		for (int i = 0; i < coat; i++) net = net + weight[Wnum + index * coat + i] * out[Onum + i];                //выходы
		out[Onum + coat + index] = 1 / (1 + exp2f(-net));
	}
}

__global__ void Delta(float* outO, float* out, float* del, int Onum, int size) {
	int index = blockIdx.x + blockIdx.y * gridDim.x;

	if (index < size)
		del[index] = (outO[index] - out[Onum + index]) * (1 - out[Onum + index]) * out[Onum + index];
}

__global__ void DeltaN(int Dnum, int Wnum, int Onum, float* del, float* weight, float* out, int coat, int n) {
	int index = blockIdx.x + blockIdx.y * gridDim.x;
	float per = 0;

	for (int i = 0; i < coat; i++)
		per = per + del[Dnum + i] * weight[Wnum + index + n * i];
	del[Dnum + coat + index] = (1 - out[Onum + index]) * out[Onum + index] * per;
}

__global__ void Deltaw(float* weight, float* del, float* out, float* delw, int Dnum, int Onum, int Wnum, int coat, int n) {
	int index = blockIdx.x + blockIdx.y * gridDim.x;
	float grad = 0;

	for (int i = 0; i < coat; i++) {
		grad = del[Dnum + i] * out[Onum + index];
		delw[Wnum + index + n * i] = 0.5 * grad + 0.3 * delw[Wnum + index + n * i];
		weight[Wnum + index + n * i] = weight[Wnum + index + n * i] + delw[Wnum + index + n * i];
	}
}

/*-------------------работает-------------------*/
