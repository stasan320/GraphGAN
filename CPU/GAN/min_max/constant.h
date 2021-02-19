#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>

typedef unsigned long long ull;

int D_max = 64;
int G_max = 64;
const int image_size = 256;
const int LayersNumD = 2;
const int LayersNumG = 4;
const float StepD = 0.25;
const float Step = 0.25;
const float Leaky = 0.05;

struct Discriminator {
	ull Onum = 0;
	ull Wnum = 0;
	ull Dnum = 0;
	ull Layer[LayersNumD] = { image_size * image_size, 1 };
};
struct Generator {
	ull Onum = 0;
	ull Wnum = 0;
	ull Dnum = 0;
	ull Layer[LayersNumG] = { 20, 20, 30, (image_size * image_size) };
};

static Discriminator discriminator;
static Generator generator;

void Random(std::vector<float>& Arr, float min, float max, ull start, ull end, clock_t MStime) {
	srand(static_cast<unsigned long long>(MStime));
	for (int i = start; i < end; i++) {
		Arr[i] = (float)(rand()) / RAND_MAX * (max - min) + min;
	}
}

void Exp(std::vector<float>& out, std::vector<float>& weight, ull* n, ull num) {
	ull Onum = 0, Wnum = 0;
	for (ull L = 0; L < (num - 1); L++) {
		for (ull i = 0; i < n[L + 1]; i++) {
			float net = 0;
			for (ull j = 0; j < n[L]; j++) {
				net = net + weight[Wnum + j + i * n[L]] * out[Onum + j];
			}
			out[Onum + n[L] + i] = 1.0 / (1.0 + exp(-net));
		}

		Onum += n[L];
		Wnum += n[L] * n[L + 1];
	}
}

void ConvExp(std::vector<float>& out, std::vector<float>& weight, ull* n, ull num) {
	ull Onum = 0, Wnum = 0;

	for (ull i = 0; i < num; i++) {
		Onum = Onum + n[i];
		Wnum = Wnum + n[i] * n[i + 1];
	}

	for (ull i = 0; i < n[num + 1]; i++) {
		float net = 0;
		for (ull j = 0; j < n[num]; j++) {
			net = net + weight[Wnum + j + i * n[num]] * out[Onum + j];
		}
		out[Onum + n[num] + i] = 1.0 / (1.0 + exp(-net));
	}
}

void Tanh(std::vector<float>& out, std::vector<float>& weight, ull* n, ull num) {
	ull Onum = 0, Wnum = 0;
	for (ull i = 0; i < num; i++) {
		Onum = Onum + n[i];
		Wnum = Wnum + n[i] * n[i + 1];
	}

	for (ull i = 0; i < n[num + 1]; i++) {
		float net = 0;
		for (ull j = 0; j < n[num]; j++) {
			net = net + weight[Wnum + j + i * n[num]] * out[Onum + j];
		}
		out[Onum + n[num] + i] = 2.0 / (1.0 + exp(-2*net)) - 1;
	}
}

void ReLu(std::vector<float>& out, std::vector<float>& weight, ull* n, ull num) {
	ull Onum = 0, Wnum = 0;
	for (ull i = 0; i < num; i++) {
		Onum = Onum + n[i];
		Wnum = Wnum + n[i] * n[i + 1];
	}

	for (ull i = 0; i < n[num + 1]; i++) {
		float net = 0;
		for (ull j = 0; j < n[num]; j++) {
			net = net + weight[Wnum + j + i * n[num]] * out[Onum + j];
		}
		//out[Onum + n[num] + i] = 1.0 / (1.0 + exp(-net));
		if (net < 0) {
			out[Onum + n[num] + i] = Leaky * net;
		}
		else {
			out[Onum + n[num] + i] = net;
		}
	}
}

void Iter(std::vector<float>& out, std::vector<float>& outOld, std::vector<float>& weight, std::vector<float>& del, std::vector<float>& delOld, ull* n, ull coat) {
	ull Onum = 0, Wnum = 0, Dnum = 0;


	for (ull i = 0; i < discriminator.Layer[LayersNumD - 2]; i++) {
		for (ull j = 0; j < discriminator.Layer[LayersNumD - 1]; j++) {
			ull OutNum = discriminator.Onum - discriminator.Layer[LayersNumD - 1] - discriminator.Layer[LayersNumD - 2];
			ull WeightNum = discriminator.Wnum - discriminator.Layer[LayersNumD - 1] * discriminator.Layer[LayersNumD - 2] + j * discriminator.Layer[LayersNumD - 2] + i;


			delOld[i] = (1 - outOld[discriminator.Onum - 1]) * outOld[OutNum + i];
			del[i] = out[discriminator.Onum - 1] * out[OutNum + i];
			weight[WeightNum] = weight[WeightNum] + StepD * (delOld[i] - del[i]);
		}
	}

	for (int L = 0; L < (coat - 2); L++) {
		for (ull i = 0; i < (coat - L - 2); i++) {
			Onum = Onum + n[i];
			Wnum = Wnum + n[i] * n[i + 1];
		}

		for (ull i = 0; i < (L); i++) {
			Dnum = Dnum + n[coat - i - 1];
		}

		for (ull i = 0; i < n[coat - L - 2]; i++) {
			float per = 0;
			for (ull j = 0; j < n[coat - L - 1]; j++) {
				ull WeightNum = Wnum - n[coat - L - 1] * n[coat - L - 2] + i + j * n[coat - L - 2];
				per += del[Dnum + j] * weight[WeightNum];
			}
			del[Dnum + n[coat - L - 1] + i] = per * (1 - out[Onum + i]) * out[Onum + i];
		}
		for (ull i = 0; i < n[coat - L - 2]; i++) {
			float per = 0;
			for (ull j = 0; j < n[coat - L - 1]; j++) {
				ull WeightNum = Wnum - n[coat - L - 1] * n[coat - L - 2] + i + j * n[coat - L - 2];
				per += delOld[Dnum + j] * weight[WeightNum];
			}
			delOld[Dnum + n[coat - L - 1] + i] = per * (1 - outOld[Onum + i]) * outOld[Onum + i];
		}

		Dnum = Dnum + n[coat - L - 1];
		Onum = 0, Wnum = 0;

		for (ull i = 0; i < (coat - L - 3); i++) {
			Onum = Onum + n[i];
			Wnum = Wnum + n[i] * n[i + 1];
		}

		for (ull i = 0; i < n[coat - L - 3]; i++) {
			for (ull j = 0; j < n[coat - L - 2]; j++) {
				ull WeightNum = Wnum + i + j * n[coat - L - 3];

				weight[WeightNum] = weight[WeightNum] + StepD * (delOld[Dnum + j] - del[Dnum + j]);
			}
		}
	}
}

void BatchTanh(cv::Mat image, std::vector<float>& out, ull Onum) {
	for (ull i = 0; i < image.rows; i++) {
		for (ull j = 0; j < image.cols; j++) {
			out[Onum + i * image.cols + j] = -1 + image.at<uchar>(i, j) / 255.0 * 2.0;
			//std::cout << out[Onum + i * image.cols + j] << "   " << image.at<uchar>(i, j)<< std::endl;
		}
	}

}

void OutTanh(std::vector<float>& out, ull Onum) {
	cv::Mat image(image_size, image_size, CV_8UC1);

	for (ull i = 0; i < image.rows; i++) {
		for (ull j = 0; j < image.cols; j++) {
			image.at<uchar>(i, j) = ceil((1 + out[Onum + j + i * image.cols]) / 2.0 * 255);
			//std::cout << (1 + out[Onum + j + i * image.cols]) / 2.0 * 255 << std::endl;
		}
	}
	//exit(0);
	cv::imshow("Out", image);
	cv::waitKey(1);
}

void BatchExp(cv::Mat image, std::vector<float>& out, ull Onum) {
	for (ull i = 0; i < image.rows; i++) {
		for (ull j = 0; j < image.cols; j++) {
			out[Onum + i * image.cols + j] = image.at<uchar>(i, j) / 255.0;
			//std::cout << out[Onum + i * image.cols + j] << "   " << image.at<uchar>(i, j)<< std::endl;
		}
	}

}

void Out(std::vector<float>& out, ull Onum) {
	cv::Mat image(image_size, image_size, CV_8UC1);

	for (ull i = 0; i < image.rows; i++) {
		for (ull j = 0; j < image.cols; j++) {
			image.at<uchar>(i, j) = out[Onum + j + i * image.cols] * 255;
		}
	}
	cv::imshow("Out", image);
	cv::waitKey(1);
}
