#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//инициализация весов
__global__ void WeightGen(float* weight, int size) {                         
	int index = blockIdx.x + blockIdx.y * gridDim.x;
	if (index < size)
		weight[index] = (exp2f(2 * net) - 1) / (exp2f(2 * net) + 1);
}

//обнуление delw
__global__ void DelwNull(float* delw, int sum) {
	int index = blockIdx.x + blockIdx.y * gridDim.x;
	if (index < sum)
		delw[index] = 0;
}

//присваивание входов
__global__ void InputData(float* data, float* out, int size) {
	int index = blockIdx.x + blockIdx.y * gridDim.x;
	if (index < size)
		out[index] = data[index];
}

//сумматор
__global__ void Sumfunc(int layer, int Wnum, int Onum, float* weight, float* out, int dop) {
	int index = blockIdx.x + blockIdx.y * gridDim.x;
	float net = 0;

	//if (index < dop) {
		for (int i = 0; i < layer; i++) {
			net = net + weight[Wnum + index * layer + i] * out[Onum + i];
		}
		//out[Onum + layer + index] = 1 / (1 + exp2f(-net));
		out[Onum + layer + index] = (exp2f(2 * net) - 1) / (exp2f(2 * net) + 1);
	//}
}

//первая дельта
__global__ void Delta(float* outO, float* out, float* del, int Onum, int size) {
	int index = blockIdx.x + blockIdx.y * gridDim.x;

	//if (index < size)
	//del[index] = (outO[index] - out[Onum + index]) * (1 - out[Onum + index]) * out[Onum + index];                                     //sigm
	del[index] = (outO[index] - out[Onum + index]) * (1 - out[Onum + index]) * (1 + out[Onum + index]);									//tang
}

//последующие дельты
__global__ void DeltaN(int Dnum, int Wnum, int Onum, float* del, float* weight, float* out, int layer, int n) {
	int index = blockIdx.x + blockIdx.y * gridDim.x;
	float per = 0;

	for (int i = 0; i < layer; i++) {
		per = per + del[Dnum + i] * weight[Wnum + index + n * i];
	}
	del[Dnum + layer + index] = (1 - out[Onum + index]) * (1 + out[Onum + index]) * per;
}

//изменение весов
__global__ void Deltaw(float* weight, float* del, float* out, float* delw, int Dnum, int Onum, int Wnum, int layer, int n) {
	int index = blockIdx.x + blockIdx.y * gridDim.x;
	float grad = 0;

	for (int i = 0; i < layer; i++) {
		grad = del[Dnum + i] * out[Onum + index];
		delw[Wnum + index + n * i] = 0.5 * grad + 0.3 * delw[Wnum + index + n * i];
		weight[Wnum + index + n * i] = weight[Wnum + index + n * i] + delw[Wnum + index + n * i];
	}
}

/*-------------------работает-------------------*/

__global__ void Clayer(float* weight, float* out, int Onum) {
	int index = blockIdx.x + blockIdx.y * gridDim.x;

	float net = 0;
	for (int i = 0; i < 16; i++)
		net = net + weight[index * 16 + i] * out[index * 16 + i];
	out[Onum + index] = (exp2f(2 * net) - 1) / (exp2f(2 * net) + 1);
}

__global__ void ConvDeltaW(float* weight, float* out, float* del, float* delw, int Dnum, int n) {
	int index = blockIdx.x + blockIdx.y * gridDim.x;
	float grad = 0;

	for (int i = 0; i < n; i++) {
		grad = del[Dnum + index] * out[index * n + i];
		delw[index * n + i] = 0.5 * grad + 0.3 * delw[index * n + i];
		weight[index * n + i] = weight[index * n + i] + delw[index * n + i];
	}
}
