#include <iostream>
#include <cmath>
#include <ctime>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "func.h"

const int coat = 4;

int main() {
	cv::Mat result(28, 28, CV_8UC1);
	int Dis[coat] = { 784, 256, 32, 1 }, Gen[coat] = {10, 32, 128, 784}, Onum = 0, Dnum = 0, Wnum = 0;
	int GenOnum = 0, GenDnum = 0, GenWnum = 0;

	clock();

	for (int i = 0; i < coat; i++) {
		Onum = Onum + Dis[i];
	}
	for (int i = 0; i < (coat - 1); i++) {
		Wnum = Wnum + Dis[i] * Dis[i + 1];
	}
	for (int i = 0; i < coat; i++) {
		GenOnum = GenOnum + Gen[i];
	}
	for (int i = 0; i < (coat - 1); i++) {
		GenWnum = GenWnum + Gen[i] * Gen[i + 1];
	}

	float* DisOut = new float[Onum];
	float* DisOutO = new float[Dis[coat - 1]];
	float* DisDel = new float[Onum];
	float* DisWeight = new float[Wnum];
	float* DisDelw = new float[Wnum];

	float* GenOut = new float[GenOnum];
	float* GenOutO = new float[Gen[coat - 1]];
	float* GenDel = new float[GenOnum];
	float* GenWeight = new float[GenWnum];
	float* GenDelw = new float[GenWnum];

	Random(DisWeight, -3, 3, 0, Wnum, clock());
	Random(GenWeight, -3, 3, 0, GenWnum, clock());
	for (int i = 0; i < Wnum; i++)
		DisDelw[i] = 0;
	for (int i = 0; i < GenWnum; i++)
		GenDelw[i] = 0;

	for (int epoch = 0; epoch < 6e4; epoch++) {
		std::cout << epoch << std::endl;
		while (DisOut[Onum - 1] > 0.5) {
		//for(int k = 0; k < 200; k++){
			cv::Mat image = cv::imread("D:\\Foton\\ngnl_data\\training\\" + std::to_string(1) + "\\1 (1).png");
			Image(image, DisOut, 0);
			for (int i = 0; i < (coat - 1); i++) {
				SumFunc(DisOut, DisWeight, Dis, i);
			}
			DisOutO[0] = 1;

			DisIterNull(DisOut, DisOutO, DisWeight, DisDelw, DisDel, Dis, coat);
			for (int i = (coat - 3); i >= 0; i--) {
				Iter(DisOut, DisWeight, DisDelw, DisDel, Dis, i, coat);
			}
			//std::cout << DisOut[Onum - 1] << std::endl;

			Random(GenOut, 0, 1, 0, Gen[0], clock());
			for (int i = 0; i < (coat - 1); i++) {
				SumFunc(GenOut, GenWeight, Gen, i);
			}
			Out(result, GenOut, (GenOnum - Gen[coat - 1]));
			for (int i = 0; i < Gen[coat - 1]; i++) {
				DisOut[i] = GenOut[GenOnum - Gen[coat - 1]];
			}
			for (int i = 0; i < (coat - 1); i++) {
				SumFunc(DisOut, DisWeight, Dis, i);
			}
			DisOutO[0] = 0;

			DisIterNull(DisOut, DisOutO, DisWeight, DisDelw, DisDel, Dis, coat);
			for (int i = (coat - 3); i >= 0; i--) {
				Iter(DisOut, DisWeight, DisDelw, DisDel, Dis, i, coat);
			}
			//std::cout << DisOut[Onum - 1] << std::endl;
		}

		while (DisOut[Onum - 1] < 0.7) {
		//for (int k = 0; k < 300; k++) {
			float index = (float)(rand()) / RAND_MAX * 784;

			Random(GenOut, -1, 1, 0, Gen[0], clock());
			for (int i = 0; i < (coat - 1); i++) {
				SumFunc(GenOut, GenWeight, Gen, i);
			}
			for (int i = 0; i < Gen[coat - 1]; i++) {
				DisOut[i] = GenOut[GenOnum - Gen[coat - 1]];
			}
			for (int i = 0; i < (coat - 1); i++) {
				SumFunc(DisOut, DisWeight, Dis, i);
			}
			float per = DisOut[Onum - 1];
			//std::cout << DisOut[Onum - 1] << std::endl;

			GenIterNull(GenOut, DisOut[Onum - 1], GenWeight, GenDelw, GenDel, Gen, coat, index, 1);
			for (int i = (coat - 3); i >= 0; i--) {
				Iter(GenOut, GenWeight, GenDelw, GenDel, Gen, i, coat);
			}

			Random(GenOut, 0, 1, 0, Gen[0], clock());
			for (int i = 0; i < (coat - 1); i++) {
				SumFunc(GenOut, GenWeight, Gen, i);
			}
			for (int i = 0; i < Gen[coat - 1]; i++) {
				DisOut[i] = GenOut[GenOnum - Gen[coat - 1]];
			}
			for (int i = 0; i < (coat - 1); i++) {
				SumFunc(DisOut, DisWeight, Dis, i);
			}

			if ((DisOut[Onum - 1] - per) <= 0) {
				GenIterNull(GenOut, DisOut[Onum - 1], GenWeight, GenDelw, GenDel, Gen, coat, index, -1.7);
				for (int i = (coat - 3); i >= 0; i--) {
					Iter(GenOut, GenWeight, GenDelw, GenDel, Gen, i, coat);
				}
			}
			Out(result, GenOut, (GenOnum - Gen[coat - 1]));
		}
	}

	return 0;
}
