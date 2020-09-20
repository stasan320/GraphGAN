#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <Windows.h>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <ctime>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "FunctionNeural.h"
#include "FunctionConsole.h"
#include "Testing.h"


const int coat = 3;

int main() {
	cv::Mat result(28, 28, CV_8UC1);
	cv::Mat image(28, 28, CV_8UC3);
	cv::Mat errors(100, 300, CV_8UC1);
	int n[coat] = { 784, 30, 10 }, Onum = 0, Dnum = 0, Wnum = 0;
	std::string path, backup_weight;
	//std::string ConfigurationPath;
	ConfigPath(path);
	//std::cout << Path;
	//return 0;
	//std::string ConfigurationPath(const std::wstring & Path);
	std::ifstream Configuration(path + "\\config.txt");
	ProgramConst(backup_weight);
	/*while(Configuration) {
		Configuration >> backup_weight;
	}*/

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

	if (backup_weight == "0") {
		Random(weight, -1.0, 1.0, 0, Wnum, clock());
	}
	else {
		std::ifstream Tweight(path + "\\weight.txt"/*, std::ios::binary | std::ios::out*/);
		for (int i = 0; i < Wnum; i++) {
			//out.write((char*)&k[i], sizeof k);
			Tweight >> weight[i];
		}
	}

	for (int i = 0; i < Wnum; i++) {
		delw[i] = 0;
	}
	for (int i = 0; i < n[coat - 1]; i++) {
		outO[i] = 0;
	}

	double error;
	int t = 0;
	unsigned int num;
	std::cout << "Enter the number of epochs: ";
	std::cin >> num;
	clock_t d = clock();

	for (unsigned int epoch = 0; epoch < num; epoch++) {
		error = 0;
		for (int k = 0; k < n[coat - 1]; k++) {
			image = cv::imread("D:\\Foton\\ngnl_data\\training\\" + std::to_string(k) + "\\1 (" + std::to_string(epoch % 500 + 1) + ").png");
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


			for (int i = 0; i < n[coat - 1]; i++) {
				error = error + (outO[i] - out[Onum - n[coat - 1] + i]) * (outO[i] - out[Onum - n[coat - 1] + i]) / n[coat - 1];
			}
			//error = error / n[coat - 1];
			outO[k] = 0.0;
		}

		if (time(NULL) > t) {
			t = time(NULL);
			std::string sec = "", min = "", hour = "";
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
			std::cout << "[" << hour << ltm->tm_hour << ":" << min << ltm->tm_min << ":" << sec << ltm->tm_sec << "][" << epoch << "][" << std::setprecision(5) << error / n[coat - 1] << "] \r";
		}
	}

	/*--------------------------Backup--------------------------*/

	std::ofstream Tweight(path + "\\weight.txt"/*, std::ios::binary | std::ios::out*/);
	for (int i = 0; i < Wnum; i++) {
		Tweight << weight[i] << " ";
	}
	std::ofstream Configurations(path + "\\config.txt");
	Configurations << "1";

	/*--------------------------Backup--------------------------*/

	std::cout << std::endl << std::endl;
	//Test(out, weight, coat, n, d);

	std::cout << std::endl;
	std::cout << "Past your path" << std::endl;

	for (;;) {
		std::string OrigName, name;
		getline(std::cin, OrigName);
		if (OrigName == "") {
			continue;
		}
		//std::cin >> name;
		for (int i = 1; i < OrigName.size() - 1; i++) {
			name += OrigName[i];
		}

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
