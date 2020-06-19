#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void Sumfunc(int coat, int Wnum, int Onum, float* weight, float* out, int dop) {
	int index = (blockIdx.x + blockIdx.y * gridDim.x) * coat;
	int net = 0;
	
	if ((blockIdx.x + blockIdx.y * gridDim.x) < dop) {
		for (int i = 0; i < coat; i++) net = net + weight[Wnum + index + i] * out[Onum + i];                //выходы
		out[Onum + coat + blockIdx.x + blockIdx.y * gridDim.x] = 1 / (1 + exp2f(-net));
	}
}

__global__ void WeightCreation(float* weight, int size) {                                        //инициализация весов
	int index = blockIdx.x + blockIdx.y * gridDim.x;
	if(index < size)
		weight[index] = 1 / (1 + exp2f(-index));
}

__global__ void InputData(float* data, float* out, int size) {
	int index = blockIdx.x + blockIdx.y * gridDim.x;
	if (index < size)
		out[index] = data[index];
}

__global__ void Delta(float* weight, float* outO, float* out, float* del, int Onum, int size) {
	int index = blockIdx.x + blockIdx.y * gridDim.x;
	if (index < size)
		del[index] = (outO[index] - out[Onum + index]) * (1 - out[Onum + index]) * out[Onum + index];
}

__global__ void Deltaw(int size, float* delw, int coat, float* del, float* out, int Wnum, int Onum, float* weight) {
	float grad;
	int index = blockIdx.x + blockIdx.y * gridDim.x;

	if (index < size) {
		grad = del[coat] * out[Onum + index];
		delw[Wnum + index] = 0.5 * grad + 0.3 * delw[Wnum + index];
		weight[Wnum + index] = weight[Wnum + index] + delw[Wnum + index];
	}
}

__global__ void DeltaN(int coat, float* del, float* weight, int Wnum, int Onum, int n, float* out) {
	float per = 0;
	int index = blockIdx.x + blockIdx.y * gridDim.x;

	for (int i = 0; i < coat; i++)
		per = per + del[Onum + coat] * weight[Wnum + i * n];
	del[Onum - n + index] = (1 - out[Onum - n + index]) * out[Onum - n + index] * per;

	/*for (i = 0; i < n[coat - 2 - l]; i++) {
				for (j = 0; j < n[coat - 1 - l]; j++) {
					ka = del[l2 + j] * weight[w - l1 + i * n[coat - 1] + j] + ka;
				}
				del[l2 + n[coat - 1 - l] + i] = (1 - out[n[coat - 2 - l] + i]) * out[l3 - n[coat - 2 - l] + i] * ka;*/
}
