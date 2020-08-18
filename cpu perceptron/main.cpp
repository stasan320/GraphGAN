#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <ctime>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "Header.h"


const int coat = 3;

int main() {
	cv::Mat result(28, 28, CV_8UC1);
	cv::Mat image(28, 28, CV_8UC3);
	cv::Mat errors(100, 300, CV_8UC1);
	int n[coat] = { 784, 55, 10 }, Onum = 0, Dnum = 0, Wnum = 0;
	int t = clock();

	for (int i = 0; i < coat; i++) {
		Onum = Onum + n[i];
	}

	for (int i = 0; i < (coat - 1); i++) {
		Wnum = Wnum + n[i] * n[i + 1];
	}
	
	double* out = new double[Onum];
	double* outO = new double[n[coat - 1]];
	double* del = new double[Onum - n[coat - 1]];
	double* weight = new double[Wnum];
	double* delw = new double[Wnum];
	std::vector<double> ErrorOut;

	Random(weight, -1.0, 1.0, 0, Wnum, clock());
	for (int i = 0; i < Wnum; i++) {
		delw[i] = 0;
	}
	for (int i = 0; i < n[coat - 1]; i++) {
		outO[i] = 0;
	}

	unsigned int num;
	std::cout << "Enter the number of epochs: ";
	std::cin >> num;
	double error;
	
	for (unsigned int epoch = 0; epoch < num; epoch++) {
		for (int k = 0; k < n[coat - 1]; k++) {
			image = cv::imread("D:\\Foton\\ngnl_data\\training\\" + std::to_string(k) + "\\1 (" + std::to_string(epoch % 5000 + 1) + ").png");
			if (!image.data) {
				std::cout << "Error upload image " << "D:\\Foton\\ngnl_data\\training\\" + std::to_string(k) + "\\1 (" + std::to_string(epoch % 5000 + 1) + ").png";
				return -1;
			}
			Image(image, out, 0);
			for (int i = 0; i < (coat - 1); i++) {
				SumFunc(out, weight, n, i);
			}
			for (int i = 0; i < n[coat - 1]; i++) {
				if (out[Onum - n[coat - 1] + i] != out[Onum - n[coat - 1] + i]) {
					std::cout << "NaN error";
					return 2;
				}
			}

			outO[k] = 1.0;
			DisIterNull(out, outO, weight, delw, del, n, coat);
			for (int i = 0; i < (coat - 2); i++) {
				Iter(out, weight, delw, del, n, i, coat);
			}

			error = 0;
			for (int i = 0; i < n[coat - 1]; i++) {
				error = error + (outO[i] - out[Onum - n[coat - 1] + i]) * (outO[i] - out[Onum - n[coat - 1] + i]);
			}
			error = error / n[coat - 1];
			ErrorOut.push_back(error);

			if (epoch % 1007 == 0) {
				for (int i = 0; i < errors.rows; i++) {
					for (int j = 0; j < errors.cols; j++) {
						errors.at<uchar>(i, j) = 0;
					}
				}

				for (int i = 0; i < ErrorOut.size(); i++) {
					float dat = (float)errors.rows - ErrorOut[i] * (float)errors.rows;
					if (dat != dat) {
						std::cout << ErrorOut[i] << std::endl;
						std::cout << error;
					}
					float min = (i * (errors.cols - 1) / (float)ErrorOut.size());
					float max = (i + 1) * (errors.cols - 1) / (float)ErrorOut.size();

					for (float j = min; j < max; j = j + 1) {
						errors.at<uchar>(round(dat), round(j)) = 255;
					}
				}
				cv::imshow("Error", errors);
				cv::waitKey(1);
			}
			//return 0;
			outO[k] = 0.0;
		}

		if (time(NULL) > t) {
			std::string sec = "", min = "", hour = "";
			t = time(NULL);
			for (int i = 0; i < n[coat - 1]; i++) {
				if (out[Onum - n[coat - 1] + i] != out[Onum - n[coat - 1] + i]) {
					std::cout << "NaN error";
					return 2;
				}
			}

			time_t now = time(NULL);
			tm* ltm = localtime(&now);
			std::cout << "                                                  \r";
			if (ltm->tm_sec < 10) {
				sec = "0";
			}
			if (ltm->tm_min < 10) {
				min = "0";
			}
			if (ltm->tm_hour < 10) {
				hour = "0";
			}
			std::cout << "[" << hour << ltm->tm_hour << ":" << min << ltm->tm_min << ":" << sec << ltm->tm_sec << "][" << epoch << "][" << std::setprecision(5) << ErrorOut[ErrorOut.size() - epoch % 7] << "] \r";
		}

		if (t == 0) {
			std::cout << std::endl;
			std::cout << "Testing" << std::endl;

			std::vector<int>DErrors(10, 0);
			for (int k = 0; k < n[coat - 1]; k++) {
				for (int test = 0; test < 890; test++) {
					image = cv::imread("D:\\Foton\\ngnl_data\\testing\\" + std::to_string(k) + "\\1 (" + std::to_string(test + 1) + ").png");
					if (!image.data) {
						std::cout << "Error upload image " << "D:\\Foton\\ngnl_data\\testing\\" + std::to_string(k) + "\\1 (" + std::to_string(test + 1) + ").png";
						return -1;
					}
					Image(image, out, 0);
					for (int i = 0; i < (coat - 1); i++) {
						SumFunc(out, weight, n, i);
					}

					double data = 0;
					int number = 0;
					for (int i = 0; i < n[coat - 1]; i++) {
						if (data < out[Onum - n[coat - 1] + i]) {
							data = out[Onum - n[coat - 1] + i];
							number = i;
						}
					}
					if (k == number) {
						DErrors[k] = DErrors[k] + 1;
					}
				}
				std::cout << "Result for " << k << ": " << std::setprecision(5) << (double)DErrors[k] / (double)900 << std::endl;
			}
			double data = 0;
			for (int i = 0; i < n[coat - 1]; i++) {
				data = DErrors[i] + data;
			}
			data = data / (900 * (double)n[coat - 1]);
			std::cout << "Result for all: " << std::setprecision(5) << data << std::endl;
		}
	}

	Testing(image, out, weight, coat, n);

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
