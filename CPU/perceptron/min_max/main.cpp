#include <iostream>
#include <cmath>
#include <vector>

#include "constant.h"

const int LayersNumD = 4;
const int LayersNumG = 4;

struct Discriminator {
	int Onum = 0;
	int Wnum = 0;
	int Dnum = 0;
	int Layer[LayersNumD] = { 784, 50, 20, 1 };
};
struct Generator {
	int Onum = 0;
	int Wnum = 0;
	int Dnum = 0;
	int Layer[LayersNumD] = { 784, 50, 20, 1 };
};

static Discriminator discriminator;
static Generator generator;

int main() {

	for (int i = 0; i < LayersNumD; i++) {
		discriminator.Onum = discriminator.Onum + discriminator.Layer[i];
	}

	for (int i = 0; i < (LayersNumD - 1); i++) {
		discriminator.Wnum = discriminator.Wnum + discriminator.Layer[i] * discriminator.Layer[i + 1];
	}

	std::vector<float> out(discriminator.Onum, 0);
	std::vector<float> outOld(discriminator.Onum, 0);
	float* outO = new float[discriminator.Layer[LayersNumD - 1]];
	//float* del = new float[Onum - n[LayersNumD - 1]] (0);
	float* weight = new float[discriminator.Wnum];

	Random(weight, -1, 1, 0, discriminator.Wnum, clock());

	/*float weight1 = 0.5;
	float d = 0.3;**/
	float* Input = new float[4];
	Input[0] = 0.5; //weigh
	Input[1] = 1;   //d
	Input[2] = 1;   //weight
	Input[3] = 0;   //d

	for (int epoch = 0; epoch < 1000; epoch++) {
		cv::Mat image = cv::imread("F:\\Foton\\ngnl_data\\training\\help\\" + std::to_string(rand() % 10) + "\\" + std::to_string(epoch % 5000) + ".png");
		Image(image, out, 0);
		for (int i = 0; i < (LayersNumD - 1); i++) {
			SumFunc(out, weight, discriminator.Layer, i);
		}

		outOld = out;
		float out1 = out[discriminator.Onum - discriminator.Layer[LayersNumD - 1]];

		image = cv::imread("F:\\Foton\\ngnl_data\\gen_image\\" + std::to_string(epoch % 5000) + ".png");
		Image(image, out, 0);
		for (int i = 0; i < (LayersNumD - 1); i++) {
			SumFunc(out, weight, discriminator.Layer, i);
		}
		float out2 = out[discriminator.Onum - discriminator.Layer[LayersNumD - 1]];


		float LossFucntion = log(out1) + log(1 - out2);
		//float LossFunctionD = ()

		std::cout << out1 << "   " << out2 << "   " << "LossD: " << LossFucntion << std::endl;

		std::vector<float>del(discriminator.Onum, 0);
		std::vector<float>delOld(discriminator.Onum, 0);
		for (int i = 0; i < discriminator.Layer[LayersNumD - 1]; i++) {
			for (int j = 0; j < discriminator.Layer[LayersNumD - 2]; j++) {
				int OutNum = discriminator.Onum - discriminator.Layer[LayersNumD - 1] - discriminator.Layer[LayersNumD - 2];
				int WeightNum = discriminator.Wnum - discriminator.Layer[LayersNumD - 1] * discriminator.Layer[LayersNumD - 2] + i * discriminator.Layer[LayersNumD - 2] + j;


				delOld[j] = (1 - out1) * outOld[OutNum + j];
				del[j] = out2 * out[OutNum + j];
				weight[WeightNum] = weight[WeightNum] + Step * (delOld[j] - del[j]);
			}
		}

		for (int i = 0; i < (LayersNumD - 2); i++) {
			Iter(out, outOld, weight, del, delOld, discriminator.Layer, i, LayersNumD);
		}
	}
}
