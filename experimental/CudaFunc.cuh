#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*-------------------global functions-------------------*/

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
	//del[Dnum + layer + index] = (1 - out[Onum + index]) * (1 + out[Onum + index]) * per;
	del[Dnum + layer + index] = (1 - out[Onum + index]) * out[Onum + index] * per;
}

__device__ void DeltaLayer(float* weight, float* del, float* out, float* delw, int Dnum, int Onum, int Wnum, int layer, int n, int ind) {
	int index = blockIdx.x + blockIdx.y * gridDim.x;
	float grad = 0;

	grad = del[Dnum + index] * out[Onum + ind];
	delw[Wnum + ind + n * index] = grad + 0.3 * delw[Wnum + ind + n * index];
	weight[Wnum + ind + n * index] = weight[Wnum + ind + n * index] + delw[Wnum + ind + n * index];
}

//изменение весов
__global__ void Deltaw(float* weight, float* del, float* out, float* delw, int Dnum, int Onum, int Wnum, int layer, int n) {
	int index = blockIdx.x + blockIdx.y * gridDim.x;
	float grad;

	for (int i = 0; i < layer; i++) {
		grad = del[Dnum + i] * out[Onum + index];
		delw[Wnum + index + n * i] = grad + 0.3 * delw[Wnum + index + n * i];
		weight[Wnum + index + n * i] = weight[Wnum + index + n * i] + delw[Wnum + index + n * i];
	}
	//DeltaLayer << <1, layer >> > (weight, del, out, delw, Dnum, Onum, Wnum, layer, n, index);
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

/*-------------------host functions-------------------*/

void Iteration(int* n, int layer, int NeuralSum, int WeightSum, float* weight, float* out, float* delw, float* Oout, float* outO, float* del) {
	int Wnum = 0, Onum = 0, Dnum = 0;
	/*cudaEvent_t t1, t2;
	cudaEventCreate(&t1);
	cudaEventCreate(&t2);

	cudaEventRecord(t1);*/
	for (int i = 0; i < (layer - 1); i++) {
		Sumfunc << <1, n[i + 1] >> > (n[i], Wnum, Onum, weight, out, n[i + 1]);
		Wnum = Wnum + n[i] * n[i + 1];
		Onum = Onum + n[i];
	}

	Onum = NeuralSum - n[layer - 1];
	Delta << <1, n[layer - 1] >> > (Oout, out, del, Onum, n[layer - 1]);
	Wnum = WeightSum;

	for (int j = 0; j < layer - 1; j++) {
		Onum = Onum - n[layer - 2 - j];
		Wnum = Wnum - n[layer - 2 - j] * n[layer - 1 - j];
		DeltaN << <1, n[layer - 2 - j] >> > (Dnum, Wnum, Onum, del, weight, out, n[layer - 1 - j], n[layer - 2 - j]);
		Dnum = Dnum + n[layer - 1 - j];
	}

	Wnum = WeightSum;
	Dnum = 0;
	Onum = NeuralSum - n[layer - 1];

	for (int j = 0; j < layer - 1; j++) {
		Onum = Onum - n[layer - 2 - j];
		Wnum = Wnum - n[layer - 1 - j] * n[layer - 2 - j];
		Deltaw << <1, n[layer - 2 - j] >> > (weight, del, out, delw, Dnum, Onum, Wnum, n[layer - 1 - j], n[layer - 2 - j]);
		Dnum = Dnum + n[layer - 1 - j];
	}
	/*cudaEventRecord(t2);
	cudaEventSynchronize(t2);
	cudaDeviceSynchronize();
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, t1, t2);
	std::cout << "CUDA time simple (ms): " << milliseconds << std::endl;
	cv::waitKey(100000);*/
}

void OutputData(int* n, int layer, float* outO, float* Oout, cv::Mat image, cv::Mat result, float* InputDataArr, float* Inp, float* out) {
	float max = 2, min = -2;
	for (int i = 0; i < n[0]; i++) {
		InputDataArr[i] = (float)(rand()) / RAND_MAX * (max - min) + min;
	}
	cudaMemcpy(Inp, InputDataArr, n[0] * sizeof(float), cudaMemcpyHostToDevice);
	InputData << <n[0], 1 >> > (Inp, out, n[0]);

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			float per = 0;
			per = image.at<cv::Vec3b>(i, j)[0];
			per = per / 255;
			outO[i * result.cols + j] = per;
		}
	}
	cudaMemcpy(Oout, outO, n[layer - 1] * sizeof(float), cudaMemcpyHostToDevice);
}

void Out(int NeuralSum, int layer, int* n, float* weights, float* out, cv::Mat image, cv::Mat result) {
	int Onum = NeuralSum - n[layer - 1];
	cudaMemcpy(weights, out, NeuralSum * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
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

void NumberInp(float* out, cv::Mat image, float* InputDataArr, int* n, int layer, float* Inp) {
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			float per = 0;
			per = image.at<cv::Vec3b>(i, j)[1];
			per = per / 255;
			InputDataArr[i * image.cols + j] = per;
		}
	}
	cudaMemcpy(Inp, InputDataArr, n[0] * sizeof(float), cudaMemcpyHostToDevice);
	InputData << < n[0], 1 >> > (Inp, out, n[0]);
}

void NumberOut(float* out, float* weights, int* n, int layer, int NeuralSum) {
	cudaMemcpy(weights, out, NeuralSum * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = NeuralSum - n[layer - 1]; i < NeuralSum; i++) {
		std::cout << weights[i] << std::endl;
	}
	std::cout << std::endl;
}

void WeightsGen(float* data, float* weight, int WeightSum) {
	float max = 2, min = -2;
	for (int i = 0; i < WeightSum; i++) {
		data[i] = (float)(rand()) / RAND_MAX * (max - min) + min;
	}
	cudaMemcpy(weight, data, WeightSum * sizeof(float), cudaMemcpyHostToDevice);
}
