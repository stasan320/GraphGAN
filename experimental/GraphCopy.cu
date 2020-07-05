#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <ctime>

#include "func.cuh"

const int layer = 3;

int main() {
	int WeightSum = 0, NeuralSum = 0, n[layer] = { 1, 20, 784 }, Wnum = 0, Onum = 0, Dnum = 0, dop = 1;
	float* del, * delw, * weight, * out, * Inp, * Oout, max = 10, min = -10;
	clock_t t1;
	std::string filename;

	for (int i = 0; i < layer; i++)
		NeuralSum = n[i] + NeuralSum;
	std::cout << "Neurals: " << NeuralSum << std::endl;

	for (int i = 0; i < layer - 1; i++)
		WeightSum = n[i] * n[i + 1] + WeightSum;
	std::cout << "Weights: " << WeightSum << std::endl;

	float* weights = new float[NeuralSum];
	float* InputDataArr = new float[n[0]];
	float* outO = new float[n[layer - 1]];

	for (int i = 0; i < n[layer - 1]; i++)
		outO[i] = 0;

	cudaMalloc((void**)&out, NeuralSum * sizeof(float));
	cudaMalloc((void**)&del, (NeuralSum - n[0]) * sizeof(float));
	cudaMalloc((void**)&weight, WeightSum * sizeof(float));
	cudaMalloc((void**)&delw, WeightSum * sizeof(float));
	cudaMalloc((void**)&Inp, n[0] * sizeof(float));
	cudaMalloc((void**)&Oout, n[layer - 1] * sizeof(float));

	WeightGen << <WeightSum, 1 >> > (weight, WeightSum);
	DelwNull << < WeightSum, 1 >> > (delw, WeightSum);

	cv::Mat image = cv::imread("E:\\Foton\\ngnl_data\\training\\0\\1 (1).png");
	/*cv::imshow("Out", image);
	cv::waitKey(10000);*/
	cv::Mat result(image.rows, image.cols, CV_8UC1);
	Input(n, layer, outO, Oout, image);
	t1 = clock();
	for (int adm = 0; adm < 1; adm++) {
		std::cout << "Iter #" << adm << std::endl;
		for (int num = 0; num < 50; num++) {
			for (int k = 0; k < 100; k++) {
				//cv::Mat image = cv::imread("E:\\Foton\\ngnl_data\\training\\" + std::to_string(k) + "\\1 (" + std::to_string(num + 1) + ").png");
				//std::cout << "E:\\Foton\\ngnl_data\\training\\" + std::to_string(k) + "\\1 (" + std::to_string(num + 1) + ").png";
				//OutputData(n, layer, outO, Oout, image, result, InputDataArr, Inp, out);
				//outO[k] = 1;
				//cudaMemcpy(Oout, outO, n[layer - 1] * sizeof(float), cudaMemcpyHostToDevice);

				//NumberInp(out, image, InputDataArr, n, layer, Inp);
				Iteration(n, layer, NeuralSum, WeightSum, weight, out, delw, Oout, outO, del);
				//NumberOut(out, weights, n, layer, NeuralSum);
				Out(NeuralSum, layer, n, weights, out, result);
				//outO[k] = 0;
				//std::cout << "Iter #" << adm << std::endl;
			}
		}
	}

	std::cout << "Time " << clock() - t1 << std::endl;

	/*for (int i = 0; i < 100; i++) {
		int Wnum = 0, Onum = 0, Dnum = 0;
		//std::cin >> filename;
		cv::Mat image = cv::imread("E:\\2.png");
		NumberInp(out, image, InputDataArr, n, layer, Inp);
		cv::imshow("Out", image);
		//Iteration(n, layer, NeuralSum, WeightSum, weight, out, delw, Oout, outO, del);
		for (int i = 0; i < (layer - 1); i++) {
			Sumfunc << <n[i + 1], 1 >> > (n[i], Wnum, Onum, weight, out, n[i + 1]);
			Wnum = Wnum + n[i] * n[i + 1];
			Onum = Onum + n[i];
		}

		NumberOut(out, weights, n, layer, NeuralSum);
	}*/
}
