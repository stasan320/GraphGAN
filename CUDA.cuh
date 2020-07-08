#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//обнуление delw
__global__ void DelwNull(float* delw, int sum) {
	int index = blockIdx.x + blockIdx.y * gridDim.x;
	if (index < sum)
		delw[index] = 0;
}

//присваивание входов
__global__ void InputData(float* data, float* out, int size, int p, int NeuralSum) {
	int index = blockIdx.x + blockIdx.y * gridDim.x;

	if (index < size)
		out[NeuralSum * p + index] = data[size * p + index];
	//out[index] = 0.542 + (exp2f(2 * index) - 1) / (exp2f(2 * index) + 1);
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
}

//первая дельта
__global__ void Delta(float* outO, float* out, float* del, int Onum, int size, int p, int n) {
	int index = blockIdx.x + blockIdx.y * gridDim.x;
	//if (index < size)
	del[index] = (outO[p * n + index] - out[Onum + index]) * (1 - out[Onum + index]) * out[Onum + index];                                     //sigm
	//del[index] = (outO[index] - out[Onum + index]) * (1 - out[Onum + index]) * (1 + out[Onum + index]);									//tang
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

//изменение весов
__global__ void Deltaw(float* weight, float* del, float* out, float* delw, int Dnum, int Onum, int Wnum, int layer, int n) {
	int index = blockIdx.x + blockIdx.y * gridDim.x;
	float grad = 0;

	for (int i = 0; i < layer; i++) {
		grad = del[Dnum + i] * out[Onum + index];
		delw[Wnum + index + n * i] = 0.5 * grad + 0.7 * delw[Wnum + index + n * i];
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
		delw[index * n + i] = grad + 0.03 * delw[index * n + i];
		weight[index * n + i] = weight[index * n + i] + delw[index * n + i];
	}
}

void Iteration(int* n, int layer, int NeuralSum, int WeightSum, float* weight, float* out, float* delw, float* Oout, float* outO, float* del) {
	int Wnum = 0, Onum = 0, Dnum = 0;

	for (int p = 0; p < 3; p++) {
		Wnum = WeightSum * p;
		Onum = NeuralSum * p;

		/*Dnum = (NeuralSum - n[0]) * p;*/
		for (int i = 0; i < (layer - 1); i++) {
			Sumfunc << <n[i + 1], 1 >> > (n[i], Wnum, Onum, weight, out, n[i + 1]);										//int layer, int Wnum, int Onum, float* weight, float* out
			Wnum = Wnum + n[i] * n[i + 1];
			Onum = Onum + n[i];
		}

		Delta << <n[layer - 1], 1 >> > (Oout, out, del, Onum, n[layer - 1], p, n[layer - 1]);

		for (int j = 0; j < layer - 1; j++) {
			Onum = Onum - n[layer - 2 - j];
			Wnum = Wnum - n[layer - 2 - j] * n[layer - 1 - j];
			DeltaN << <n[layer - 2 - j], 1 >> > (Dnum, Wnum, Onum, del, weight, out, n[layer - 1 - j], n[layer - 2 - j]);					    //int Dnum, int Wnum, int Onum, float* del, float* weight, float* out
			Dnum = Dnum + n[layer - 1 - j];
		}

		Wnum = WeightSum * (p + 1);
		Dnum = 0;
		Onum = NeuralSum * (p + 1) - n[layer - 1];

		for (int j = 0; j < layer - 1; j++) {
			Onum = Onum - n[layer - 2 - j];
			Wnum = Wnum - n[layer - 1 - j] * n[layer - 2 - j];
			Deltaw << < n[layer - 2 - j], 1 >> > (weight, del, out, delw, Dnum, Onum, Wnum, n[layer - 1 - j], n[layer - 2 - j]);				//float* weight, float* del, float* out, float* delw, int Dnum, int Onum, int Wnum, int layer, int n
			Dnum = Dnum + n[layer - 1 - j];
		}
	}
}

void Input(int n, float* outO, float* Oout, cv::Mat image) {
	for (int p = 0; p < 3; p++) {
		for (int i = 0; i < image.rows; i++) {
			for (int j = 0; j < image.cols; j++) {
				float per = 0;
				per = image.at<cv::Vec3b>(i, j)[p];
				per = per / 255;
				outO[p * image.cols * image.rows + i * image.cols + j] = per;
			}
		}
	}
	cudaMemcpy(Oout, outO, n * 3 * sizeof(float), cudaMemcpyHostToDevice);
}

void Out(int NeuralSum, int layer, int* n, float* out, cv::Mat result) {
	float* weights = new float[NeuralSum * 3];
	cudaMemcpy(weights, out, NeuralSum * 3 * sizeof(float), cudaMemcpyDeviceToHost);

	for (int p = 0; p < 3; p++) {
		int Onum = NeuralSum * (p + 1) - n[layer - 1];
		for (int i = 0; i < result.rows; i++) {
			for (int j = 0; j < result.cols; j++) {
				float per = 0;
				per = weights[Onum + i * result.cols + j];
				per = ceil(per * 255);
				//std::cout << per << std::endl;
				result.at<cv::Vec3b>(i,j)[p] = per;
			}
		}
	}
	cv::imshow("Out", result);
	cv::waitKey(1);
	delete[] weights;
}

//инициализация весов
void WeightGen(float* weight, float* delw, int WeightSum) {
	float* data = new float[WeightSum * 3];
	float max = 3, min = -3;
	srand(static_cast<unsigned int>(time(0)));

	for (int i = 0; i < WeightSum * 3; i++) {
		data[i] = (float)(rand()) / RAND_MAX * (max - min) + min;
	}
	cudaMemcpy(weight, data, WeightSum * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(delw, data, WeightSum * 3 * sizeof(float), cudaMemcpyHostToDevice);
	delete[] data;
}

//инициализация входов
void InputGen(int n, int NeuralSum, float* Inp, float* out) {
	float* data = new float[n * 3];
	float max = 1, min = -1;
	srand(static_cast<unsigned int>(time(0)));

	for (int i = 0; i < n; i++) {
		data[i] = (float)(rand()) / RAND_MAX * (max - min) + min;
	}
	cudaMemcpy(Inp, data, n * 3 * sizeof(float), cudaMemcpyHostToDevice);
	for (int i = 0; i < 3; i++)
		InputData << <n, 1 >> > (Inp, out, n, i, NeuralSum);
	delete[] data;
}

void DataCheck(int WeightSum, float* weight, float* delw) {
	std::string filename;

	std::ifstream Fconfig("E:\\Foton\\ngnl_data\\backup\\config.txt");
	Fconfig >> filename;
	if (std::stoi(filename) == 1) {
		Fconfig.close();
		float* Bweight = new float[WeightSum * 3];
		float* Bdelw = new float[WeightSum * 3];
		std::ifstream fweight("E:\\Foton\\ngnl_data\\backup\\weight.dat");
		std::ifstream fdelw("E:\\Foton\\ngnl_data\\backup\\delw.dat");

		for (int i = 0; i < WeightSum * 3; i++) {
			fweight >> Bweight[i];
			fdelw >> Bdelw[i];
		}
		std::cout << "Model upload" << std::endl;

		cudaMemcpy(weight, Bweight, WeightSum * 3 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(delw, Bdelw, WeightSum * 3 * sizeof(float), cudaMemcpyHostToDevice);
		delete[] Bweight;
		delete[] Bdelw;
	}
	else if (std::stoi(filename) == 0) {
		WeightGen(weight, delw, WeightSum);
	}
}

void InputImage(int n, float* out, float* outO, float* Oout, float* Inp, cv::Mat image) {
	int p = 0;
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			float per = 0;
			per = image.at<cv::Vec3b>(i, j)[0];
			per = per / 255;
			outO[i * image.cols + j] = per;
		}
	}

	cudaMemcpy(Oout, outO, n * sizeof(float), cudaMemcpyHostToDevice);
	InputData << <n, 1 >> > (Inp, out, n, p, p);
}

void ImageResult(int NeuralSum, float* out, int n) {
	float* weights = new float[NeuralSum];
	cudaMemcpy(weights, out, NeuralSum * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = (NeuralSum - n); i < NeuralSum; i++)
		std::cout << weights[i] << std::endl;
}
