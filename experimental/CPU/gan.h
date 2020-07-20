#include <windows.h>
#include <iostream>
#include <cmath>
#include <time.h>
#include "Header.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

using namespace std;

const int layer = 4;


int main() {
	int  w = 0, n[layer] = { 784, 512, 32, 1 }, kl, nc[layer] = { 16, 128, 512, 784 };
	float iter = 1, per, Giter = 1;
	int min = -3, max = 3, DNeuralSum = 0, GNeuralSum = 0, DWeightSum = 0, GWeightSum = 0;

	for (int i = 0; i < layer; i++) {
		DNeuralSum = n[i] + DNeuralSum;
	}
	for (int i = 0; i < layer - 1; i++) {
		DWeightSum = n[i] * n[i + 1] + DWeightSum;
	}

	for (int i = 0; i < layer; i++) {
		GNeuralSum = nc[i] + GNeuralSum;
	}
	for (int i = 0; i < layer - 1; i++) {
		GWeightSum = nc[i] * nc[i + 1] + GWeightSum;
	}

	float* outO = new float[n[layer - 1]];
	float* del = new float[DNeuralSum - n[0]];
	float* out = new float[DNeuralSum];
	float* weight = new float[DWeightSum];
	float* delw = new float[DWeightSum];

	float* GoutO = new float[nc[layer - 1]];
	float* Gdel = new float[GNeuralSum - nc[0]];
	float* Gout = new float[GNeuralSum];
	float* Gweight = new float[GWeightSum];
	float* Gdelw = new float[GWeightSum];

	srand(static_cast<unsigned int>(clock()));
	for (int i = 0; i < GWeightSum; i++) {
		Gweight[i] = (float)(rand()) / RAND_MAX * (max - min) + min;
		Gdelw[i] = 0;
	}

	srand(static_cast<unsigned int>(clock()));
	for (int i = 0; i < DWeightSum; i++) {
		weight[i] = (float)(rand()) / RAND_MAX * (max - min) + min;
		delw[i] = 0;
	}


	GoutO[0] = 0;
	GoutO[1] = 0;


	/*outO[2] = 0.6;
	outO[3] = 0.785;
	outO[4] = 1;*/
	cv::Mat image = cv::imread("E:\\Foton\\ngnl_data\\training\\" + std::to_string(1) + "\\1 (" + std::to_string(1) + ").png");
	cv::Mat result(image.rows, image.cols, CV_8UC1);

	cv::imshow("Out", image);
	cv::waitKey(1);

	for (int k = 0; k < 100000; k++) {
		std::cout << "Iter #" << k << std::endl;
		for (int i = 0; i < 300; i++) {
			//Iter random data//
			Random(Gout, nc[0]);
			SumG(Gweight, Gout, nc, layer);
			for (int j = 0; j < n[0]; j++) {
				out[j] = Gout[GNeuralSum - nc[layer - 1] + j];
				//std::cout << out[j] << endl;
			}
			SumD(weight, out, n, layer);
			//Out(out, n);
			//std::cout << std::endl;
			outO[0] = 0;
			DisIter(del, outO, out, weight, delw, n, iter, layer, DNeuralSum, DWeightSum);
			//iter = iter * 0.999;

			//}
			//Iter true data//
			for (int j = 0; j < 1; j++) {
				cv::Mat image = cv::imread("E:\\Foton\\ngnl_data\\training\\" + std::to_string(1) + "\\1 (" + std::to_string(1) + ").png");
				for (int l = 0; l < image.rows; l++) {
					for (int ln = 0; ln < image.cols; ln++) {
						float per = 0;
						per = image.at<cv::Vec3b>(l, ln)[0];
						per = per / 255;
						out[l * image.cols + ln] = per;
						//std::cout << outO[i * image.cols + j] << std::endl;
					}
				}
				SumD(weight, out, n, layer);
				//if ((out[n[0] + n[1]] > 0.8) && (k > 5)) {
				outO[0] = 1;
				DisIter(del, outO, out, weight, delw, n, iter, layer, DNeuralSum, DWeightSum);
				//iter = iter * 0.999;
			}
			//Out(out, n);
			//std::cout << std::endl;
		}

		for (int i = 0; i < n[0]; i++) {
			for (int j = 0; j < 1; j++) {
				//round 1
				Random(Gout, nc[0]);
				SumG(Gweight, Gout, nc, layer);
				//Out(Gout, nc);
				for (int l = 0; l < n[0]; l++) {
					out[l] = Gout[GNeuralSum - n[0] + j];
					//std::cout << Gout[GNeuralSum - n[0] + j] << std::endl;
				}
				for (int l = 0; l < result.rows; l++) {
					for (int ln = 0; ln < result.cols; ln++) {
						float per = 0;
						per = Gout[GNeuralSum - nc[layer - 1] + l * result.cols + ln];
						per = per * 255;
						per = ceil(per);
						//std::cout << per << std::endl;
						result.at<uchar>(l, ln) = per;
					}
				}
				cv::imshow("Out", result);
				cv::waitKey(1);
				//std::cout << endl;
				SumD(weight, out, n, layer);
				Out(out, n);
				//Out(out, n);
				//std::cout << std::endl;
				for (int l = 0; l < n[layer - 1]; l++) {
					GoutO[1] = GoutO[0];
					GoutO[0] = out[DNeuralSum - n[layer - 1] + l];
					//std::cout << GoutO[0] - GoutO[1] << std::endl;
				}
				per = 1;
				GenIter(Gdel, GoutO, Gout, Gweight, Gdelw, nc, i, per, Giter, layer, GWeightSum, GNeuralSum);

				//round 2
				Random(Gout, nc[0]);
				SumG(Gweight, Gout, nc, layer);
				for (int l = 0; l < n[0]; l++) {
					out[l] = Gout[GNeuralSum - nc[layer - 1] + j];
				}
				SumD(weight, out, n, layer);
				for (int l = 0; l < n[layer - 1]; l++) {
					GoutO[1] = GoutO[0];
					GoutO[0] = out[DNeuralSum - n[layer - 1] + l];
					//std::cout << GoutO[0] - GoutO[1] << std::endl;
				}
				if ((GoutO[0] - GoutO[1]) <= 0) {
					per = -2;
					GenIter(Gdel, GoutO, Gout, Gweight, Gdelw, nc, i, per, Giter, layer, GWeightSum, GNeuralSum);
				}
			}
			//Giter = Giter * 0.999;
		}
	}
	return 0;
}
