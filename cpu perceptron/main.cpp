//Create by Stasan
//nan-error ? вероятно из-за нарушения работы loss function

//1, 2 - 784, 32, 4, 2						//
//3, 4 - 784, 64, 16, 3						//sigm
//5, 6, 7, 8 - 784, 128, 16, 5				//sigm
//9 ?										//

//2 - 784, 32, 4, 2                         //tang, в основном работает, иногда вылетает nan, испр проверку на NaN


#include <iostream>
#include <cmath>
#include <ctime>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "Header.h"

const int coat = 4;

int main() {
	cv::Mat result(28, 28, CV_8UC1);
	int n[coat] = { 784, 32, 4, 2 }, Onum = 0, Dnum = 0, Wnum = 0;
	int t = clock();

	for (int i = 0; i < coat; i++) {
		Onum = Onum + n[i];
	}

	for (int i = 0; i < (coat - 1); i++) {
		Wnum = Wnum + n[i] * n[i + 1];
	}

	float* out = new float[Onum];
	float* outO = new float[n[coat - 1]];
	float* del = new float[Onum];
	float* weight = new float[Wnum];
	float* delw = new float[Wnum];

	Random(weight, -1, 1, 0, Wnum, clock());
	for (int i = 0; i < Wnum; i++) {
		delw[i] = 0;
	}
	for (int i = 0; i < n[coat - 1]; i++)
		outO[i] = 0;

	for (int epoch = 0; epoch < 6000; epoch++) {
		for (int k = 0; k < n[coat - 1]; k++) {
			cv::Mat image = cv::imread("D:\\Foton\\ngnl_data\\training\\" + std::to_string(k) + "\\1 (" + std::to_string(epoch % 100 + 1) + ").png");
			if (!image.data) {
				std::cout << "Error upload image";
				return -1;
			}
			cv::imshow("Out", image);
			cv::waitKey(1);
			Image(image, out, 0);
			for (int i = 0; i < (coat - 1); i++) {
				SumFunc(out, weight, n, i);
				//std::cout << std::endl;
			}
			//return 0;
			//Out(result, out, Onum - n[coat - 1]);
			for (int i = 0; i < n[coat - 1]; i++) {
				//std::cout << out[Onum - n[coat - 1] + i] << std::endl;
			}
			if (out[Onum - n[coat - 1]] == NAN) {
				std::cout << "NaN error";
				return 2;
			}

			outO[k] = 1;
			DisIterNull(out, outO, weight, delw, del, n, coat);
			for (int i = 0; i < (coat - 2); i++) {
				Iter(out, weight, delw, del, n, i, coat);
			}
			//return 0;
			outO[k] = -1;
			//std::cout << std::endl;
		}
	}
	for (;;) {
		std::string name;
		getline(std::cin, name);
		cv::Mat image = cv::imread(name);
		if (!image.data) {
			std::cout << "Error upload image";
			continue;
		}
		cv::imshow("Out", image);
		cv::waitKey(1);
		Image(image, out, 0);
		for (int i = 0; i < (coat - 1); i++) {
			SumFunc(out, weight, n, i);
			//std::cout << std::endl;
		}
		//return 0;
		//Out(result, out, Onum - n[coat - 1]);
		for (int i = 0; i < n[coat - 1]; i++) {
			std::cout << out[Onum - n[coat - 1] + i] << std::endl;
		}
	}
	return 0;
}
