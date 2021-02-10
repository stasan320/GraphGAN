#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>

#include "Header.h"

const int LayersNumD = 2;
const int LayersNumG = 2;

struct Discriminator {
	int Onum = 0;
	int Wnum = 0;
	int Dnum = 0;
	int Layer[LayersNumD] = { 784, 1 };
};
struct Generator {
	int Onum = 0;
	int Wnum = 0;
	int Dnum = 0;
	int Layer[LayersNumD] = { 20, 784 };
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

	for (int i = 0; i < LayersNumG; i++) {
		generator.Onum = generator.Onum + generator.Layer[i];
	}
	for (int i = 0; i < (LayersNumG - 1); i++) {
		generator.Wnum = generator.Wnum + generator.Layer[i] * generator.Layer[i + 1];
	}

	std::vector<float> out(discriminator.Onum, 0);
	std::vector<float> outOld(discriminator.Onum, 0);
	std::vector<float> weight(discriminator.Wnum);
	std::vector<float>del(discriminator.Onum, 0);
	std::vector<float>delOld(discriminator.Onum, 0);

	std::vector<float> Gout(generator.Onum, 0);
	std::vector<float> GoutOld(generator.Onum, 0);
	std::vector<float> Gweight(generator.Wnum);
	std::vector<float> Gdel(discriminator.Onum, 0);
	std::vector<float> GdelOld(discriminator.Onum, 0);

	Random(weight, -1, 1, 0, discriminator.Wnum, clock());
	Random(Gweight, -1, 1, 0, generator.Wnum, clock());

	int dop = 0;

	for (int epoch = 0; epoch < 10000; epoch++) {
		for (int D = 0; D < 10; D++) {
			cv::Mat image = cv::imread("G:\\Foton\\data\\cropped\\1 (" + std::to_string((epoch + dop) % 5000 + 1) + ").jpg", CV_8UC1);
			if (!image.data) {
				dop++;
				image = cv::imread("G:\\Foton\\data\\cropped\\1 (" + std::to_string((epoch + dop) % 5000 + 1) + ").jpg", CV_8UC1);
			}

			cv::resize(image, image, cv::Size(28, 28));
			Image(image, out, 0);

			for (int i = 0; i < (LayersNumD - 1); i++) {
				SumFunc(out, weight, discriminator.Layer, i);
			}

			outOld = out;
			float out1 = out[discriminator.Onum - discriminator.Layer[LayersNumD - 1]];

			Random(Gout, 0, 1, 0, generator.Layer[0], clock());
			for (int i = 0; i < (LayersNumG - 1); i++) {
				SumFunc(Gout, Gweight, generator.Layer, i);
			}
			for (int i = 0; i < generator.Layer[LayersNumG - 1]; i++) {
				out[i] = Gout[generator.Onum - generator.Layer[LayersNumG - 1] + i];
			}
			//Out(out, 0);


			for (int i = 0; i < (LayersNumD - 1); i++) {
				SumFunc(out, weight, discriminator.Layer, i);
			}
			float out2 = out[discriminator.Onum - discriminator.Layer[LayersNumD - 1]];


			float LossFucntionD = log(out1) + log(1 - out2);
			//float LossFucntionG = log(out2);
			//float LossFunctionD = ()

			//std::cout.precision(3);
			//std::cout << out1 << "   " << out2 << "   " << "LossD: " << LossFucntionD << std::endl;


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


		for (int G = 0; G < 160; G++) {
			Random(Gout, 0, 1, 0, generator.Layer[0], clock());
			for (int i = 0; i < (LayersNumG - 1); i++) {
				SumFunc(Gout, Gweight, generator.Layer, i);
			}
			for (int i = 0; i < generator.Layer[LayersNumG - 1]; i++) {
				out[i] = Gout[generator.Onum - generator.Layer[LayersNumG - 1] + i];
			}
			Out(out, 0);


			for (int i = 0; i < (LayersNumD - 1); i++) {
				SumFunc(out, weight, discriminator.Layer, i);
			}
			float out2 = out[discriminator.Onum - discriminator.Layer[LayersNumD - 1]];
			float LossFucntionG = log(out2);
			std::cout << out2 << "   " << "LossG: " << LossFucntionG << std::endl;

			for (int i = 0; i < discriminator.Layer[LayersNumD - 1]; i++) {
				for (int j = 0; j < discriminator.Layer[LayersNumD - 2]; j++) {
					int WeightNum = discriminator.Wnum - discriminator.Layer[LayersNumD - 2] * discriminator.Layer[LayersNumD - 1] + j + i * discriminator.Layer[LayersNumD - 1];
					Gdel[j] = (1 - out2) * out[j] * (1 - out[j]) * weight[WeightNum];
				}
			}

			for (int i = 0; i < generator.Layer[LayersNumG - 1]; i++) {
				for (int j = 0; j < generator.Layer[LayersNumG - 2]; j++) {
					int WeightNum = generator.Wnum - generator.Layer[LayersNumG - 2] * generator.Layer[LayersNumG - 1] + j + i * generator.Layer[LayersNumG - 2];
					int OutNum = generator.Onum - generator.Layer[LayersNumG - 2] - generator.Layer[LayersNumG - 1] + j;
					//std::cout << WeightNum << std::endl;
					Gweight[WeightNum] = Gweight[WeightNum] + Step * Gdel[i] * Gout[OutNum];
				}
			}
		}
	}
}
