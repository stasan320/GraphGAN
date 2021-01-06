#define _CRT_SECURE_NO_WARNINGS

#ifdef WIN32
#define ourImread(filename, isColor) cvLoadImage(filename.c_str(), isColor)
#else
#define ourImread(filename, isColor) imread(filename, isColor)
#endif

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
	int n[coat] = { 784, 10, 1 }, Onum = 0, Dnum = 0, Wnum = 0;
	int g[coat] = { 6, 10, 784 }, Onumg = 0, Dnumg = 0, Wnumg = 0;
	float step = 0.36;
	float old[2] = { 0, 0 };
	//std::string path, ProgramData[2];
	//path = ConfigPath();
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

	for (int i = 0; i < coat; i++) {
		Onumg = Onumg + g[i];
	}

	for (int i = 0; i < (coat - 1); i++) {
		Wnumg = Wnumg + g[i] * g[i + 1];
	}

	float* out = new float[Onum];
	float* outO = new float[n[coat - 1]];
	float* del = new float[Onum - n[coat - 1]];
	float* weight = new float[Wnum];
	float* delw = new float[Wnum];
	std::vector<float> ErrorOut;

	float* Gout = new float[Onumg];
	float* GoutO = new float[g[coat - 1]];
	float* Gdel = new float[Onumg - n[coat - 1]];
	float* Gweight = new float[Wnumg];
	float* Gdelw = new float[Wnumg];

	Random(weight, -1, 1, 0, Wnum, clock());
	Random(Gweight, -1, 1, 0, Wnumg, clock());
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

		//Discriminator(0, n[coat - 1], epoch, coat, step, Onum, n, weight, out, delw, del, error);
		for (int k = 0; k < 1; k++) {
			std::cout << "D:\\Foton\\ngnl_data\\training\\help\\" + std::to_string(k) + "\\" + std::to_string(epoch % 5000) + ".png" << std::endl;
			cv::Mat image = cv::imread("D:\\Foton\\ngnl_data\\training\\help\\" + std::to_string(k) + "\\" + std::to_string(epoch % 5000) + ".png", CV_LOAD_IMAGE_ANYDEPTH);         //на k > 2500 и image > +-5000 на порядок падает скорость
			//IplImage* image= cvLoadImage(path, CV_LOAD_IMAGE_UNCHANGED);
			//image(image);
			if (!image.data) {
				continue;
			}
			Image(image, out, 0);
			cv::imshow("Out1", image);
			cv::waitKey(1);

			for (int i = 0; i < (coat - 1); i++) {
				SumFunc(out, weight, n, i);
			}
			std::cout << out[Onum - 1] << std::endl;


			outO[k] = 1;
			DisIterNull(out, outO, weight, delw, del, n, coat, step);
			for (int i = 0; i < (coat - 2); i++) {
				Iter(out, weight, delw, del, n, i, coat, step);
			}


			for (int i = 0; i < n[coat - 1]; i++) {
				error = error + (outO[i] - out[Onum - n[coat - 1] + i]) * (outO[i] - out[Onum - n[coat - 1] + i]) / n[coat - 1];
			}
			outO[k] = 0;


			/*-------------------------------------------------------------------------------*/

			srand(static_cast<unsigned long long>(clock()));
			for (int i = 0; i < g[0]; i++) {
				Gout[i] = (float)rand() / RAND_MAX;
			}

			//сумматор
			for (int i = 0; i < (coat - 1); i++) {
				SumFunc(Gout, Gweight, g, i);

			}


			//вывод
			for (int i = 0; i < 28; i++) {
				for (int j = 0; j < 28; j++) {
					out[i * 28 + j] = Gout[Onumg - g[coat - 1] + i * 28 + j];
				}
			}

			for (int i = 0; i < (coat - 1); i++) {
				SumFunc(out, weight, n, i);

			}
			std::cout << out[Onum - 1] << std::endl << std::endl;
			outO[k] = 0;

			DisIterNull(out, outO, weight, delw, del, n, coat, step);
			for (int i = 0; i < (coat - 2); i++) {
				Iter(out, weight, delw, del, n, i, coat, step);
			}


			for (int i = 0; i < n[coat - 1]; i++) {
				error = error + (outO[i] - out[Onum - n[coat - 1] + i]) * (outO[i] - out[Onum - n[coat - 1] + i]) / n[coat - 1];
			}
			outO[k] = 0;
		}


		for (int dl = 0; dl < 30; dl++) {
			old[0] = 1;
			srand(static_cast<unsigned long long>(clock()));
			int N = ceil(((float)rand() / RAND_MAX) * g[coat - 1]);
			//srand(static_cast<clock_t>(clock() + 1));

			int test = 3;
			float delt = 1;

			srand(static_cast<unsigned long long>(clock()));
			//std::cout << N << std::endl;
			for (int l = 0; l < test; l++) {
				for (int k = 0; k < 1; k++) {
					//N = k;
					//srand(static_cast<unsigned long long>(clock()));
					for (int i = 0; i < g[0]; i++) {
						Gout[i] = (float)rand() / RAND_MAX;
					}

					//шум на вход

					//сумматор
					for (int i = 0; i < (coat - 1); i++) {
						SumFunc(Gout, Gweight, g, i);
					}

					/*-------------------------------------------------------------------------------*/
					//вывод
					for (int i = 0; i < 28; i++) {
						for (int j = 0; j < 28; j++) {
							out[i * 28 + j] = Gout[Onumg - g[coat - 1] + i * 28 + j];
						}
					}

					Out(out, 0);
					/*-------------------------------------------------------------------------------*/

					//cv::Mat image = cv::imread("D:\\Foton\\ngnl_data\\training\\" + std::to_string(k) + "\\1 (" + std::to_string(epoch % 3000 + 1) + ").png");         //на k > 2500 и image > +-5000 на порядок падает скорость
					//Image(image, out, 0);

					for (int i = 0; i < (coat - 1); i++) {
						SumFunc(out, weight, n, i);
					}


					old[1] = out[Onum - 1];
					//std::cout << old[0] << "    " << old[1] << std::endl;
					if (old[1] < old[0]) {
						
						delt = delt * (-1);
					}
					//std::cout << delt << std::endl;
					old[0] = old[1];
					//std::cout << old[1] << std::endl;



					/*cv::Mat image = cv::imread("D:\\Foton\\ngnl_data\\training\\" + std::to_string(k) + "\\1 (" + std::to_string(epoch % 3000 + 1) + ").png");
					Image(image, GoutO, 0);*/

					GenIterNull(Gout, GoutO, Gweight, Gdelw, Gdel, g, coat, step, old, N, l, delt);
					for (int i = 0; i < (coat - 2); i++) {
						Iter(Gout, Gweight, Gdelw, Gdel, g, i, coat, step);
					}
				}

				//консоль
				if (time(NULL) > t) {
					//std::cout << std::endl << clock() - d  << std::endl;        //время выполнения цикла k
					t = time(NULL);
					ConsoleData(d, epoch, error, n, coat);
				}
			}
		}


		//консоль
		if (time(NULL) > t) {
			//std::cout << std::endl << clock() - d  << std::endl;        //время выполнения цикла k
			t = time(NULL);
			ConsoleData(d, epoch, error, n, coat);
		}

	}

	std::cout << std::endl;
	std::cout << "fdssdfsfdsf";
	/*for (int k = 0; k < 100000; k++) {
		srand(static_cast<unsigned long long>(clock() + time(0)));
		for (int i = 0; i < g[0]; i++) {
			Gout[i] = (float)rand() / RAND_MAX;
		}

		//сумматор
		for (int i = 0; i < (coat - 1); i++) {
			SumFunc(Gout, Gweight, g, i);
		}

		for (int i = 0; i < 28; i++) {
			for (int j = 0; j < 28; j++) {
				out[i * 28 + j] = Gout[Onumg - g[coat - 1] + i * 28 + j];
			}
		}

		Out(out, 0);
		std::cout << "sd" << std::endl;
	}*/
	std::cout << std::endl << std::endl;
	GlobalTime(clock() - d);
	Testing(out, weight, coat, n);
	

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
