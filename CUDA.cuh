#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//инициализация весов
__global__ void WeightGen(float* weight, int size) {                         
	int index = blockIdx.x + blockIdx.y * gridDim.x;
	if (index < size)
		weight[index] = 1 / (1 + exp2f(index) / size);
		//weight[index] = 1 / (1 + exp2f(-index));
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
		out[index] = 0.542 + (exp2f(2 * index) - 1) / (exp2f(2 * index) + 1);
}

//сумматор
__global__ void Sumfunc(int layer, int Wnum, int Onum, float* weight, float* out, int dop) {
	int index = blockIdx.x + blockIdx.y * gridDim.x;
	float net = 0;

	//if (index < dop) {
		for (int i = 0; i < layer; i++) {
			net = net + weight[Wnum + index * layer + i] * out[Onum + i];
		}
		out[Onum + layer + index] = 1 / (1 + exp2f(-net));
		//out[Onum + layer + index] = (exp2f(2 * net) - 1) / (exp2f(2 * net) + 1);
	//}
}

//первая дельта
__global__ void Delta(float* outO, float* out, float* del, int Onum, int size) {
	int index = blockIdx.x + blockIdx.y * gridDim.x;

	//if (index < size)
	del[index] = (outO[index] - out[Onum + index]) * (1 - out[Onum + index]) * out[Onum + index];                                     //sigm
	//del[index] = (outO[index] - out[Onum + index]) * (1 - out[Onum + index]) * (1 + out[Onum + index]);									//tang
}

//последующие дельты
__global__ void DeltaN(int Dnum, int Wnum, int Onum, float* del, float* weight, float* out, int layer, int n) {
	int index = blockIdx.x + blockIdx.y * gridDim.x;
	float per = 0;

	for (int i = 0; i < layer; i++) {
		per = per + del[Dnum + i] * weight[Wnum + index + n * i];
	}
	del[Dnum + layer + index] = (1 - out[Onum + index]) * (1 + out[Onum + index]) * per;
	//del[Dnum + layer + index] = (1 - out[Onum + index]) * out[Onum + index] * per;
}

//изменение весов
__global__ void Deltaw(float* weight, float* del, float* out, float* delw, int Dnum, int Onum, int Wnum, int layer, int n) {
	int index = blockIdx.x + blockIdx.y * gridDim.x;
	float grad = 0;

	for (int i = 0; i < layer; i++) {
		grad = del[Dnum + i] * out[Onum + index];
		delw[Wnum + index + n * i] = grad + 0.3 * delw[Wnum + index + n * i];
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

void Iteration(int* n, int layer, int NeuralSum, int WeightSum, float* weight, float* out, float* delw, float* Oout, float* outO, float* del) {
	int Wnum = 0, Onum = 0, Dnum = 0;

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
}

void Input(int* n, int layer, float* outO, float* Oout, cv::Mat image) {
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			float per = 0;
			per = image.at<cv::Vec3b>(i, j)[0];
			per = per / 255;
			outO[i * image.cols + j] = per;
		}
	}
	cudaMemcpy(Oout, outO, n[layer - 1] * sizeof(float), cudaMemcpyHostToDevice);
}

void Out(int NeuralSum, int layer, int* n, float* weights, float* out, cv::Mat result) {
	int Onum = NeuralSum - n[layer - 1];
	cudaMemcpy(weights, out, NeuralSum * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < result.rows; i++) {
		for (int j = 0; j < result.cols; j++) {
			float per = 0;
			per = weights[Onum + i * result.cols + j];
			per = per * 255;
			per = ceil(per);
			//std::cout << per << std::endl;
			result.at<uchar>(i, j) = per;
		}
	}
	cv::imshow("Out", result);
	cv::waitKey(1);
}
