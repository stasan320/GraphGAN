#include <iostream>
#include <cmath>
#include <vector>

#include "constant.h"


const int LayersNum = 3;

int main() {
	int n[LayersNum] = { 784, 8, 1 }, Onum = 0, Dnum = 0, Wnum = 0;
	//std::cout << Path;
	//return 0;
	//std::string ConfigurationPath(const std::wstring & Path);
	/*std::ifstream Configuration(path + "\\config");
	ProgramConst(ProgramData);
	step = std::stod(ProgramData[1]);*/

	/*while(Configuration) {
		Configuration >> backup_weight;
	}*/

	for (int i = 0; i < LayersNum; i++) {
		Onum = Onum + n[i];
	}

	for (int i = 0; i < (LayersNum - 1); i++) {
		Wnum = Wnum + n[i] * n[i + 1];
	}

	std::vector<float> out(Onum, 0);
	std::vector<float> outOld(Onum, 0);
	float* outO = new float[n[LayersNum - 1]];
	//float* del = new float[Onum - n[LayersNum - 1]] (0);
	float* weight = new float[Wnum];
	float* delw = new float[Wnum];

	Random(weight, -1, 1, 0, Wnum, clock());

	/*float weight1 = 0.5;
	float d = 0.3;**/
	float* Input = new float[4];
	Input[0] = 0.5; //weigh
	Input[1] = 1;   //d
	Input[2] = 1;   //weight
	Input[3] = 0;   //d

	for (int epoch = 0; epoch < 10000; epoch++) {
		cv::Mat image = cv::imread("F:\\Foton\\ngnl_data\\training\\help\\" + std::to_string(rand() % 10) + "\\" + std::to_string(epoch % 5000) + ".png");
		Image(image, out, 0);
		for (int i = 0; i < (LayersNum - 1); i++) {
			SumFunc(out, weight, n, i);
		}

		outOld = out;
		float out1 = out[Onum - n[LayersNum - 1]];

		image = cv::imread("F:\\Foton\\ngnl_data\\gen_image\\" + std::to_string(epoch % 1000) + ".png");
		Image(image, out, 0);
		for (int i = 0; i < (LayersNum - 1); i++) {
			SumFunc(out, weight, n, i);
		}
		float out2 = out[Onum - n[LayersNum - 1]];


		float LossFucntion = log(out1) + log(1 - out2);
		//float LossFunctionD = ()

		std::cout << out1 << "   " << out2 << "   " << LossFucntion << std::endl;

		std::vector<float>del(Onum, 0);
		for (int i = 0; i < n[LayersNum - 1]; i++) {
			for (int j = 0; j < n[LayersNum - 2]; j++) {
				int OutNum = Onum - n[LayersNum - 1] - n[LayersNum - 2];
				int WeightNum = Wnum - n[LayersNum - 1] * n[LayersNum - 2] + i * n[LayersNum - 2] + j;


				float delta = (1 - out1) * outOld[OutNum + j] - out2 * out[OutNum + j];
				weight[WeightNum] = weight[WeightNum] + Step * delta;

				del[i] = del[i] + delta;
			}
		}

		for (int i = 0; i < n[LayersNum - 1]; i++) {
			//std::cout << del[i] << std::endl;
		}

		for (int i = 0; i < (LayersNum - 2); i++) {
			//Iter(out, weight, delw, del, n, i, LayersNum);
		}
		return 0;
	}
}
