#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

int D_max = 64;
int G_max = 64;
const int image_size = 256;
const int LayersNumD = 2;
const int LayersNumG = 3;
const float Step = 0.7;
const float Leaky = 0;

struct Discriminator {
	int Onum = 0;
	int Wnum = 0;
	int Dnum = 0;
	int Layer[LayersNumD] = { image_size * image_size, 1 };
};
struct Generator {
	int Onum = 0;
	int Wnum = 0;
	int Dnum = 0;
	int Layer[LayersNumG] = { 30, 40, image_size * image_size };
};

static Discriminator discriminator;
static Generator generator;

void Random(std::vector<float>& Arr, float min, float max, int start, int end, clock_t MStime) {
	srand(static_cast<unsigned long long>(MStime + time(0)));
	for (int i = start; i < end; i++) {
		Arr[i] = (float)(rand()) / RAND_MAX * (max - min) + min;
	}
}

void Exp(std::vector<float>& out, std::vector<float>& weight, int* n, int num) {
	int Onum = 0, Wnum = 0;
	for (int i = 0; i < num; i++) {
		Onum = Onum + n[i];
		Wnum = Wnum + n[i] * n[i + 1];
	}

	for (int i = 0; i < n[num + 1]; i++) {
		float net = 0;
		for (int j = 0; j < n[num]; j++) {
			net = net + weight[Wnum + j + i * n[num]] * out[Onum + j];
		}
		out[Onum + n[num] + i] = 1.0 / (1.0 + exp(-net));
	}
}

void ReLu(std::vector<float>& out, std::vector<float>& weight, int* n, int num) {
	int Onum = 0, Wnum = 0;
	for (int i = 0; i < num; i++) {
		Onum = Onum + n[i];
		Wnum = Wnum + n[i] * n[i + 1];
	}

	for (int i = 0; i < n[num + 1]; i++) {
		float net = 0;
		for (int j = 0; j < n[num]; j++) {
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

void Iter(std::vector<float>& out, std::vector<float>& outOld, std::vector<float>& weight, std::vector<float>& del, std::vector<float>& delOld, int* n, int num, int coat) {
	int Onum = 0, Wnum = 0, Dnum = 0;
	for (int i = 0; i < (coat - num - 2); i++) {
		Onum = Onum + n[i];
		Wnum = Wnum + n[i] * n[i + 1];
	}

	for (int i = 0; i < (num); i++) {
		Dnum = Dnum + n[coat - i - 1];
	}

	for (int i = 0; i < n[coat - num - 2]; i++) {
		float per = 0;
		for (int j = 0; j < n[coat - num - 1]; j++) {
			per += del[Dnum + j] / (float)n[coat - num - 1];
		}
		del[Dnum + n[coat - num - 1] + i] = per * (1 - out[Onum + i]) * out[Onum + i];
	}
	for (int i = 0; i < n[coat - num - 2]; i++) {
		float per = 0;
		for (int j = 0; j < n[coat - num - 1]; j++) {
			per += delOld[Dnum + j] / (float)n[coat - num - 1];
		}
		delOld[Dnum + n[coat - num - 1] + i] = per * (1 - outOld[Onum + i]) * outOld[Onum + i];
	}

	Dnum = Dnum + n[coat - num - 1];
	Onum = 0, Wnum = 0;

	for (int i = 0; i < (coat - num - 3); i++) {
		Onum = Onum + n[i];
		Wnum = Wnum + n[i] * n[i + 1];
	}

	for (int i = 0; i < n[coat - num - 2]; i++) {
		for (int j = 0; j < n[coat - num - 3]; j++) {
			int WeightNum = Wnum + j + i * n[coat - num - 3];

			weight[WeightNum] = weight[WeightNum] + Step * (delOld[Dnum + j] - del[Dnum + j]);
		}
	}
}

void Image(cv::Mat image, std::vector<float>& out, int Onum) {
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			out[Onum + i * image.cols + j] = image.at<uchar>(i, j) / (float)255;
			//std::cout << out[Onum + i * image.cols + j] << "   " << image.at<uchar>(i, j)<< std::endl;
		}
	}

}

void Out(std::vector<float>& out, int Onum) {
	cv::Mat image(image_size, image_size, CV_8UC1);

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			image.at<unsigned char>(i, j) = out[Onum + j + i * image.cols] * 255;
		}
	}
	cv::imshow("Out", image);
	cv::waitKey(1);
}
