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
	int Layer[LayersNumD] = { image_size * image_size, 1 };
};
struct Generator {
	int Onum = 0;
	int Wnum = 0;
	int Dnum = 0;
	int Layer[LayersNumG] = { 20, image_size * image_size };
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

	//Random(Gout, 0, 1, 0, generator.Layer[0], clock());


	for (int epoch = 0; epoch < 10000; epoch++) {
		for (int D = 0; D < D_max; D++) {
			cv::Mat image = cv::imread("G:\\Foton\\data\\cropped\\1 (" + std::to_string((epoch * D_max + D + dop) % 5000 + 1) + ").jpg", CV_8UC1);
			if (!image.data) {
				dop++;
				image = cv::imread("G:\\Foton\\data\\cropped\\1 (" + std::to_string((epoch * D_max + D + dop) % 5000 + 1) + ").jpg", CV_8UC1);
			}
			//cv::Mat image = cv::imread("C:\\Users\\stasan\\Downloads\\archive\\mnist_png\\testing\\" + std::to_string(rand() % 10) + "\\1 (" + std::to_string((epoch * D_max + D) % 890 + 1) + ").png", CV_8UC1);


			cv::resize(image, image, cv::Size(image_size, image_size));
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


		for (int G = 0; G < G_max; G++) {
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

			int WeightDopNum = 0, OutDopNum = 0, DelDopNum = 0;
			WeightDopNum += discriminator.Layer[LayersNumD - 2] * discriminator.Layer[LayersNumD - 1];
			OutDopNum += discriminator.Layer[LayersNumD - 1];

			for (int i = 0; i < discriminator.Layer[LayersNumD - 2]; i++) {
				float per = 0;
				for (int j = 0; j < discriminator.Layer[LayersNumD - 1]; j++) {
					int WeightNum = discriminator.Wnum - WeightDopNum + j + i * discriminator.Layer[LayersNumD - 1];
					int OutNum = discriminator.Onum - OutDopNum + j;

					per += (1 - out[OutNum]) * weight[WeightNum];
				}
				int OutNum = discriminator.Onum - OutDopNum - discriminator.Layer[LayersNumD - 2] + i;
				Gdel[DelDopNum + i] = per * out[OutNum] * (1 - out[OutNum]);
			}

			for (int num = 0; num < (LayersNumD - 2); num++) {
				WeightDopNum += discriminator.Layer[LayersNumD - 2 - num] * discriminator.Layer[LayersNumD - 1 - num];
				OutDopNum += discriminator.Layer[LayersNumD - 1 - num];
				DelDopNum += discriminator.Layer[LayersNumD - 2 - num];

				for (int i = 0; i < discriminator.Layer[LayersNumD - 2 - num]; i++) {
					float per = 0;
					for (int j = 0; j < discriminator.Layer[LayersNumD - 1 - num]; j++) {
						int WeightNum = discriminator.Wnum - WeightDopNum + j + i * discriminator.Layer[LayersNumD - 1 - num];
						int OutNum = discriminator.Onum - OutDopNum + j;

						//per += 
					}
					Gdel[DelDopNum + i] = per * out[i] * (1 - out[i]);
				}
				//DelDopNum += 
			}

			for (int i = 0; i < generator.Layer[LayersNumG - 2]; i++) {
				for (int j = 0; j < generator.Layer[LayersNumG - 1]; j++) {
					int WeightNum = generator.Wnum - generator.Layer[LayersNumG - 2] * generator.Layer[LayersNumG - 1] + i + j * generator.Layer[LayersNumG - 2];
					int OutNum = generator.Onum - generator.Layer[LayersNumG - 2] - generator.Layer[LayersNumG - 1] + i;
					//std::cout << WeightNum << std::endl;
					Gweight[WeightNum] = Gweight[WeightNum] + Step * Gdel[DelDopNum + j] * Gout[OutNum];
				}
			}
		}
	}
}
