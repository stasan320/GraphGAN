#include <iostream>
#include <cmath>
#include <ctime>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "func.h"

const int coat = 3;

int main() {
	cv::Mat result(28, 28, CV_8UC1);
	int n[coat] = { 1, 2, 784 }, Onum = 0, Dnum = 0, Wnum = 0;
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

	int time = clock();

	Random(weight, -3, 3, 0, Wnum, time);
	for (int i = 0; i < Wnum; i++)
		delw[i] = 0;

	for (int epoch = 0; epoch < 6e4; epoch++) {
			cv::Mat image = cv::imread("D:\\Foton\\ngnl_data\\training\\" + std::to_string(1) + "\\1 (" + std::to_string(1) + ").png");
			out[0] = 1;
			for (int i = 0; i < (coat - 1); i++) {
				SumFunc(out, weight, n, i);
			}
			Out(result, out, Onum - n[coat - 1]);

			Image(image, outO, 0);
			IterNull(out, outO, weight, delw, del, n, coat);
			for (int i = (coat - 3); i >= 0; i--) {
				Iter(out, weight, delw, del, n, i, coat);
			}
	}

	return 0;
}
