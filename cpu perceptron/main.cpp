#include <iostream>
#include <cmath>
#include <ctime>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "Header.h"

const int coat = 3;

int main() {
	cv::Mat result(28, 28, CV_8UC1);
	cv::Mat image(28, 28, CV_8UC1);
	int n[coat] = { 784, 32, 2 }, Onum = 0, Dnum = 0, Wnum = 0;
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

	Random(weight, -1, 3, 0, Wnum, clock());
	for (int i = 0; i < Wnum; i++)
		delw[i] = 0.5;
	for (int i = 0; i < n[coat - 1]; i++)
		outO[i] = 0;

	for (int epoch = 0; epoch < 6e4; epoch++) {
		for (int k = 0; k < n[coat - 1]; k++) {
			image = cv::imread("D:\\Foton\\ngnl_data\\training\\" + std::to_string(k) + "\\1 (1).png");
			if (!image.data) {
				std::cout << "Error upload image" << std::endl;
				return -1;
			}
			cv::imshow("Out", image);
			cv::waitKey(1);
			Image(image, out, 0);
			for (int i = 0; i < (coat - 1); i++) {
				SumFunc(out, weight, n, i);
			}
			//Out(result, out, Onum - n[coat - 1]);
			//if (epoch % 40 == 0) {
				for (int i = 0; i < n[coat - 1]; i++) {
					std::cout << out[Onum - n[coat - 1] + i] << std::endl;
				}
			//}

			outO[k] = 1;
			DisIterNull(out, outO, weight, delw, del, n, coat);
			for (int i = 0; i < (coat - 2); i++) {
				Iter(out, weight, delw, del, n, i, coat);
			}
			//return 0;
			outO[k] = 0;
			std::cout << std::endl;
		}
	}

	return 0;
}

