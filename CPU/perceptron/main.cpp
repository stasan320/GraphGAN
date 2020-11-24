#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <Windows.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <ctime>

//#include <boost/lexical_cast.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "FunctionNeural.h"
#include "FunctionConsole.h"
#include "FunctionTesting.h"
#include "OldFunc.h"

const int coat = 3;

int main() {
	cv::Mat result(28, 28, CV_8UC1);
	cv::Mat image(28, 28, CV_8UC3);
	cv::Mat errors(100, 300, CV_8UC1);
	int n[coat] = { 784, 30, 10 }, Onum = 0, Dnum = 0, Wnum = 0;
	float step = 0.36;
	std::string path, ProgramData[2];
	path = ConfigPath();
	//std::cout << Path;
	//return 0;
	//std::string ConfigurationPath(const std::wstring & Path);
	/*std::ifstream Configuration(path + "\\config");
	ProgramConst(ProgramData);
	step = std::stod(ProgramData[1]);*/

	/*while(Configuration) {
		Configuration >> backup_weight;
	}*/

	for (int i = 0; i < coat; i++) {
		Onum = Onum + n[i];
	}

	for (int i = 0; i < (coat - 1); i++) {
		Wnum = Wnum + n[i] * n[i + 1];
	}

	float* out = new float[Onum];
	float* outO = new float[n[coat - 1]];
	float* del = new float[Onum - n[(int)coat - (int)1]];
	float* weight = new float[Wnum];
	float* delw = new float[Wnum];
	std::vector<float> ErrorOut;

	Random(weight, -1, 1, 0, Wnum, clock());
	/*if (ProgramData[0] == "0") {
		Random(weight, -1, 1, 0, Wnum, clock());
	}
	else {
		std::ifstream Tweight(path + "\\weight");
		//std::ofstream w(path + "\\weight");
		for (int i = 0; i < Wnum; i++) {
			//out.write((char*)&k[i], sizeof k);
			Tweight >> weight[i];
			//w << weight[i] << " ";
		}
	}*/

	for (int i = 0; i < Wnum; i++) {
		delw[i] = 0;
	}
	for (int i = 0; i < n[coat - 1]; i++) {
		outO[i] = 0;
	}

	float error;
	time_t t = 0;
	unsigned int num;
	std::cout << "Enter the number of epochs: ";
	std::cin >> num;
	clock_t d = clock();

	for (unsigned int epoch = 0; epoch < num; epoch++) {
		error = 0;
		for (int k = 0; k < n[coat - 1]; k++) {
			image = cv::imread("D:\\Foton\\ngnl_data\\training\\" + std::to_string(k) + "\\1 (" + std::to_string(epoch % 5000 + 1) + ").png");
			Image(image, out, 0);
			for (int i = 0; i < (coat - 1); i++) {
				SumFunc(out, weight, n, i);
			}

			outO[k] = 1.0;
			DisIterNull(out, outO, weight, delw, del, n, coat, step);
			for (int i = 0; i < (coat - 2); i++) {
				Iter(out, weight, delw, del, n, i, coat, step);
			}


			for (int i = 0; i < n[coat - 1]; i++) {
				error = error + (outO[i] - out[Onum - n[coat - 1] + i]) * (outO[i] - out[Onum - n[coat - 1] + i]) / n[coat - 1];
			}
			outO[k] = 0.0;
		}

		 //консоль
		if (time(NULL) > t) {
			t = time(NULL);
			Time(d, epoch, error, n, coat);
		}
	}

	std::cout << std::endl << std::endl;
	GlobalTime(clock() - d);
	Testing(image, out, weight, coat, n);
	

	/*--------------------------Backup--------------------------*/

	//BackupWeight(path, ProgramData, Wnum, weight);

	/*--------------------------Backup--------------------------*/


	std::cout << std::endl;
	std::cout << "Past your path" << std::endl;


	//ручное тестирование
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
