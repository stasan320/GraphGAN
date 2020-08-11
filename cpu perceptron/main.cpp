#include <iostream>
#include <cmath>
#include <ctime>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "Header.h"

const int coat = 5;

int main() {
	cv::Mat result(28, 28, CV_8UC1);
	cv::Mat image(28, 28, CV_8UC3);
	int n[coat] = { 784, 512, 128, 64, 10 }, Onum = 0, Dnum = 0, Wnum = 0;
	int t = clock();

	for (int i = 0; i < coat; i++) {
		Onum = Onum + n[i];
	}

	for (int i = 0; i < (coat - 1); i++) {
		Wnum = Wnum + n[i] * n[i + 1];
	}
	
	double* out = new double[Onum];
	double* outO = new double[n[coat - 1]];
	double* del = new double[Onum];
	double* weight = new double[Wnum];
	double* delw = new double[Wnum];

	Random(weight, -3, 3, 0, Wnum, clock());
	for (int i = 0; i < Wnum; i++) {
		delw[i] = 0;
	}
	for (int i = 0; i < n[coat - 1]; i++) {
		outO[i] = 0;
	}
	std::cout << "Start" << std::endl;
	int num;
	std::cout << "Enter the number of epochs ";
	std::cin >> num;
	for (unsigned int epoch = 0; epoch < num; epoch++) {
		for (int k = 0; k < n[coat - 1]; k++) {
			image = cv::imread("D:\\Foton\\ngnl_data\\training\\" + std::to_string(k) + "\\1 (" + std::to_string(epoch % 5000 + 1) + ").png");
			if (!image.data) {
				std::cout << "Error upload image";
				return -1;
			}
			Image(image, out, 0);
			for (int i = 0; i < (coat - 1); i++) {
				SumFunc(out, weight, n, i);
			}

			for (int i = 0; i < n[coat - 1]; i++) {
				if (out[Onum - n[coat - 1]] != out[Onum - n[coat - 1]]) {
					std::cout << "NaN error";
					return 2;
				}
			}

			outO[k] = 1;
			DisIterNull(out, outO, weight, delw, del, n, coat);
			for (int i = 0; i < (coat - 2); i++) {
				Iter(out, weight, delw, del, n, i, coat);
			}
			outO[k] = 0;
		}
		if (epoch % 50 == 0) {
			double error = 0;
			for (int i = 0; i < n[coat - 1]; i++) {
				error = error + (outO[i] - out[Onum - n[coat - 1] + i]) * (outO[i] - out[Onum - n[coat - 1] + i]) / n[coat - 1];
			}
			std::cout << "Epoch[" << epoch << "][" << error << "] \r";
		}
	}
	std::cout << std::endl;
	std::cout << "Past your path" << std::endl;

	for (;;) {
		std::string name;
		getline(std::cin, name);
		cv::Mat image = cv::imread(name);
		if (!image.data) {
			std::cout << "Error upload image" << std::endl;
			continue;
		}
		cv::imshow("Out", image);
		cv::waitKey(1);
		Image(image, out, 0);
		for (int i = 0; i < (coat - 1); i++) {
			SumFunc(out, weight, n, i);
		}
		for (int i = 0; i < n[coat - 1]; i++) {
			if (out[Onum - n[coat - 1]] != out[Onum - n[coat - 1]]) {
				std::cout << "NaN error";
				return 2;
			}
		}

		double data = 0;
		int number = 0;
		for (int i = 0; i < n[coat - 1]; i++) {
			if (data < out[Onum - n[coat - 1] + i]) {
				data = out[Onum - n[coat - 1] + i];
				number = i;
			}
			std::cout << i << " - " << out[Onum - n[coat - 1] + i] << std::endl;
		}
		std::cout << "It's " << number << std::endl;
	}
	return 0;
}
