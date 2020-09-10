#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//обнуление delw
__global__ void Null(float* delw, int sum) {
	int index = blockIdx.x + blockIdx.y * gridDim.x;
	if (index < sum)
		delw[index] = 0;
}

//присваивание входов
__global__ void InputData(float* data, float* out, int size, int p, int NeuralSum) {
	int index = blockIdx.x + blockIdx.y * gridDim.x;

	//if (index < size)
		out[NeuralSum * p + index] = data[index];
	//out[index] = 0.542 + (exp2f(2 * index) - 1) / (exp2f(2 * index) + 1);
}

__global__ void InputDataF(float* data, float* out, int size) {
	int index = blockIdx.x + blockIdx.y * gridDim.x;

	if (index < size)
		out[index] = data[index];
	//out[index] = 0.542 + (exp2f(2 * index) - 1) / (exp2f(2 * index) + 1);
}


__global__ void InputDataArr(float* data, float* out, int DisNeuralSum, int p, int NeuralSum, int size) {
	int index = blockIdx.x + blockIdx.y * gridDim.x;

	//if (index < size)
		out[NeuralSum * p + index] = data[DisNeuralSum * (p + 1) - size + index];
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
__global__ void DisDelta(float* outO, float* out, float* del, int Onum, int size, int p) {
	int index = blockIdx.x + blockIdx.y * gridDim.x;

	//if (index < size)
	del[index] = (outO[index] - out[Onum + index]) * (1 - out[Onum + index]) * out[Onum + index];                                     //sigm
	//del[index] = (outO[index] - out[Onum + index]) * (1 - out[Onum + index]) * (1 + out[Onum + index]);									//tang
}

__global__ void GenDelta(float* outO, float* out, float* del, int Onum, int size, int p, int RGB, float var) {
	int index = blockIdx.x + blockIdx.y * gridDim.x;
	//if (index < size)
	//del[index] = (log10f(outO[p]) - out[Onum + index]) * (1 - out[Onum + index]) * out[Onum + index];                                     //sigm
	//del[index] = var * log2f(1 - outO[p]) * (1 - out[Onum + index]) * out[Onum + index];
	//del[index] = (outO[index] - out[Onum + index]) * (1 - out[Onum + index]) * (1 + out[Onum + index]);									//tang


	del[index] = var * log2f(1 - outO[p]) * (1 - out[Onum + index]) * out[Onum + index];
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
		delw[Wnum + index + n * i] = 0.5 * grad + 0.002 * delw[Wnum + index + n * i];
		weight[Wnum + index + n * i] = weight[Wnum + index + n * i] + delw[Wnum + index + n * i];
	}
}

//инициализация весов
void WeightGen(float* weight, float* delw, int WeightSum, int RGB) {
	float* data = new float[WeightSum * RGB];
	float max = 3, min = -3;
	srand(static_cast<unsigned int>(time(0)));

	for (int i = 0; i < WeightSum * RGB; i++) {
		data[i] = (float)(rand()) / RAND_MAX * (max - min) + min;
	}
	cudaMemcpy(weight, data, WeightSum * RGB * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(delw, data, WeightSum * RGB * sizeof(float), cudaMemcpyHostToDevice);
	delete[] data;
}

//рандомайзер
void Random(int n, int NeuralSum, float* Inp, float* out, int RGB) {
	float* data = new float[n * RGB];
	float max = 1, min = -1;
	srand(static_cast<unsigned int>(clock()));

	for (int i = 0; i < n * RGB; i++) {
		data[i] = (float)(rand()) / RAND_MAX * (max - min) + min;
		//std::cout << data[i] << std::endl;
	}
	cudaMemcpy(Inp, data, n * RGB * sizeof(float), cudaMemcpyHostToDevice);
	for (int i = 0; i < RGB; i++)
		InputData << <n, 1 >> > (Inp, out, n, i, NeuralSum);
	delete[] data;
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

/*-------------------работает-------------------*/

void DataCheck(int WeightSum, float* weight, float* delw, int p, int RGB) {
	std::string filename;

	std::ifstream Fconfig("E:\\Foton\\ngnl_data\\backup\\config" + std::to_string(p) + ".txt");
	Fconfig >> filename;
	if (std::stoi(filename) == 1) {
		Fconfig.close();
		float* Bweight = new float[WeightSum * RGB];
		float* Bdelw = new float[WeightSum * RGB];
		std::ifstream fweight("E:\\Foton\\ngnl_data\\backup\\weight" + std::to_string(p) + ".dat");
		std::ifstream fdelw("E:\\Foton\\ngnl_data\\backup\\delw" + std::to_string(p) + ".dat");

		for (int i = 0; i < WeightSum * RGB; i++) {
			fweight >> Bweight[i];
			fdelw >> Bdelw[i];
		}
		std::cout << "Model upload" << std::endl;

		cudaMemcpy(weight, Bweight, WeightSum * RGB * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(delw, Bdelw, WeightSum * RGB * sizeof(float), cudaMemcpyHostToDevice);
		delete[] Bweight;
		delete[] Bdelw;
	}
	else if (std::stoi(filename) == 0) {
		WeightGen(weight, delw, WeightSum, RGB);
	}
}

void Backup(int WeightSum, float* weight, float* delw, int p, int RGB) {
	float* Bweight = new float[WeightSum * RGB];
	float* Bdelw = new float[WeightSum * RGB];
	cudaMemcpy(Bweight, weight, WeightSum * RGB * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(Bdelw, delw, WeightSum * RGB * sizeof(float), cudaMemcpyDeviceToHost);
	std::ofstream fweight("E:\\Foton\\ngnl_data\\backup\\weight" + std::to_string(p) + ".dat");
	std::ofstream fdelw("E:\\Foton\\ngnl_data\\backup\\delw" + std::to_string(p) + ".dat");

	for (int i = 0; i < WeightSum * RGB; i++) {
		fweight << Bweight[i] << " ";
		fdelw << Bdelw[i] << " ";
	}
	std::cout << "Backup" + std::to_string(p) << std::endl;
	delete[] Bweight;
	delete[] Bdelw;
	fweight.close();
	fdelw.close();

	std::ofstream config("E:\\Foton\\ngnl_data\\backup\\config" + std::to_string(p) + ".txt");
	config << 1;
	config.close();
}

/*---------------neural void func---------------*/

void GlobalSumFunc(int* n, int layer, int NeuralSum, int WeightSum, float* weight, float* out, int RGB) {
	int Wnum, Onum;

	for (int p = 0; p < RGB; p++) {
		Wnum = WeightSum * p;
		Onum = NeuralSum * p;

		for (int i = 0; i < (layer - 1); i++) {
			Sumfunc << <n[i + 1], 1 >> > (n[i], Wnum, Onum, weight, out, n[i + 1]);										//int layer, int Wnum, int Onum, float* weight, float* out
			Wnum = Wnum + n[i] * n[i + 1];
			Onum = Onum + n[i];
		}
	}
}

void DisIteration(int* n, int layer, int NeuralSum, int WeightSum, float* weight, float* out, float* delw, float* Oout, float* outO, float* del, int RGB) {
	int Wnum, Onum, Dnum = 0;

	for (int p = 0; p < RGB; p++) {
		Wnum = WeightSum * (p + 1);
		Onum = NeuralSum * (p + 1) - n[layer - 1];

		DisDelta << <n[layer - 1], 1 >> > (Oout, out, del, Onum, n[layer - 1], p);

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

void GenIteration(int* n, int layer, int NeuralSum, int WeightSum, float* weight, float* out, float* delw, float* Oout, float* outO, float* del, int RGB, float var) {
	int Wnum = 0, Onum = 0, Dnum = 0;

	for (int p = 0; p < RGB; p++) {
		Wnum = WeightSum * (p + 1);
		Onum = NeuralSum * (p + 1) - n[layer - 1];

		GenDelta << <n[layer - 1], 1 >> > (Oout, out, del, Onum, n[layer - 1], p, RGB, var);

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


void InputOutImage(int n, float* outO, float* Oout, cv::Mat image, int RGB) {
	for (int p = 0; p < RGB; p++) {
		for (int i = 0; i < image.rows; i++) {
			for (int j = 0; j < image.cols; j++) {
				float per = 0;
				per = image.at<cv::Vec3b>(i, j)[p];
				per = per / 255;
				outO[p * image.cols * image.rows + i * image.cols + j] = per;
				//std::cout << outO[p * image.cols * image.rows + i * image.cols + j] << std::endl;
			}
		}
	}
	cudaMemcpy(Oout, outO, n * RGB * sizeof(float), cudaMemcpyHostToDevice);
}

void OutOutImage(int NeuralSum, int layer, int* n, float* out, cv::Mat image, int RGB) {
	float* weights = new float[NeuralSum * RGB];
	cudaMemcpy(weights, out, NeuralSum * RGB * sizeof(float), cudaMemcpyDeviceToHost);

	for (int p = 0; p < RGB; p++) {
		int Onum = NeuralSum * (p + 1) - n[layer - 1];
		for (int i = 0; i < image.rows; i++) {
			for (int j = 0; j < image.cols; j++) {
				float per = 0;
				per = weights[Onum + i * image.cols + j];
				per = ceil(per * 255);
				//std::cout << per << std::endl;
				image.at<uchar>(i, j) = per;
			}
		}
	}
	cv::imshow("Out", image);
	cv::waitKey(1);
	delete[] weights;
}

void InputInputImage(int n, float* out, float* outO, float* Inp, cv::Mat image, int DisNeuralSum, int RGB) {
	float* weights = new float[n * RGB];
	for (int p = 0; p < RGB; p++) {
		for (int i = 0; i < image.rows; i++) {
			for (int j = 0; j < image.cols; j++) {
				float per = 0;
				per = image.at<cv::Vec3b>(i, j)[p];
				per = per / 255;
				weights[i * image.cols + j] = per;
				//std::cout << outO[i * image.cols + j] << std::endl;
			}
		}
	}

	cudaMemcpy(Inp, weights, n * RGB * sizeof(float), cudaMemcpyHostToDevice);
	for (int i = 0; i < RGB; i++)
		InputData << <n, 1 >> > (Inp, out, n, 0, 0);
	delete[] weights;
}

void ImageResult(int NeuralSum, float* out, int n, int RGB) {
	float* weights = new float[NeuralSum * RGB];
	cudaMemcpy(weights, out, NeuralSum * RGB * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < RGB; i++)
		std::cout << weights[(i + 1) * NeuralSum - n] << std::endl;
	delete[] weights;
}

void ImageOpt(int NeuralSum, float* out, int n, float* outO, int RGB) {
	float* weights = new float[NeuralSum * RGB];
	cudaMemcpy(weights, out, NeuralSum * RGB * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < RGB; i++)
		outO[i] = weights[(i + 1) * NeuralSum - n];
	delete[] weights;
}

void InputDiffer(int i, int* n, int layer, int WeightSum, int nc, int DisNeuralSum, int NeuralSum, float* DisInp, float* Disout, float* weight, float* out, float* outO, int p, float* Inp, int RGB) {
	if (i == 0) {
		Random(nc, DisNeuralSum, DisInp, Disout, RGB);
		//i = 1;
	}
	else {
		Random(n[0], NeuralSum, Inp, out, RGB);
		GlobalSumFunc(n, layer, NeuralSum, WeightSum, weight, out, RGB);

		for (int i = 0; i < RGB; i++)
			InputDataArr << <nc, 1 >> > (out, Disout, DisNeuralSum, i, NeuralSum, nc);
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
	DisDelta << <n[layer - 1], 1 >> > (Oout, out, del, Onum, n[layer - 1], 3);
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

void Out(int NeuralSum, int layer, int* n, float* weight, float* out, cv::Mat result) {
	int Onum = NeuralSum - n[layer - 1];
	float* weights = new float[NeuralSum];
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
	delete[] weights;
}

void OptDis(float* DisoutO, int nc, int RGB, float* DisOout, int p) {
	for (int i = 0; i < nc * RGB; i++)
		DisoutO[i] = p;
	cudaMemcpy(DisOout, DisoutO, nc * RGB * sizeof(float), cudaMemcpyHostToDevice);
}

void Convert(float* Inp, int n, int NeuralSum, int DisNeuralSum, float* Disout, float* out, int RGB) {
	float* data = new float[n * RGB];
	float* weights = new float[NeuralSum * RGB];

	cudaMemcpy(weights, out, NeuralSum * RGB * sizeof(float), cudaMemcpyDeviceToHost);
	for (int j = 0; j < RGB; j++) {
		for (int i = 0; i < n; i++) {
			data[i + n * j] = weights[NeuralSum * (j + 1) - n + i];
		}
	}

	cudaMemcpy(Inp, data, n * RGB * sizeof(float), cudaMemcpyHostToDevice);
	for (int i = 0; i < RGB; i++)
		InputData << <n, 1 >> > (Inp, Disout, n, i, DisNeuralSum);
	delete[] weights;
	delete[] data;
}

void ConvertDis(float* Inp, int n, int NeuralSum, int DisNeuralSum, float* Disout, int RGB, float* var, float* Vvar) {
	float* weights = new float[DisNeuralSum * RGB];
	float* data = new float[RGB];
	cudaMemcpy(weights, Disout, DisNeuralSum * RGB * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(data, Inp, RGB * sizeof(float), cudaMemcpyDeviceToHost);
	//std::cout << data[0] << " 1" << std::endl;
	Vvar[0] = data[0];
	for (int i = 0; i < RGB; i++) {
		data[i] = weights[(i + 1) * DisNeuralSum - n];
		//std::cout << data[i] << " 0" << std::endl;
	}
	Vvar[0] = data[0] - Vvar[0];
	if (Vvar[0] < 0)
		Vvar[0] = -1;
	else if(Vvar > 0) {
		Vvar[0] = 1;
	}
	cudaMemcpy(Inp, data, RGB * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(var, Vvar, RGB * sizeof(float), cudaMemcpyHostToDevice);
	delete[] weights;
	delete[] data;
}
