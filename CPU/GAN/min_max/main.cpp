#include "constant.h"
#include <Windows.h>

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
	std::vector<float> outReal(discriminator.Onum, 0);
	std::vector<float> weight(discriminator.Wnum);
	std::vector<float> del(discriminator.Onum, 0);
	std::vector<float> delOld(discriminator.Onum, 0);

	std::vector<float> Gout(generator.Onum, 0);
	std::vector<float> GoutOld(generator.Onum, 0);
	std::vector<float> Gweight(generator.Wnum);
	std::vector<float> Gdel(discriminator.Onum + generator.Onum, 0);

	Random(weight, -1, 1, 0, discriminator.Wnum, clock());
	Random(Gweight, -1, 1, 0, generator.Wnum, clock());

	int dop = 0;

	//Random(Gout, 0, 1, 0, generator.Layer[0], clock());


	for (ull epoch = 0; epoch < 10000; epoch++) {
		for (ull D = 0; D < DMAX; D++) {
			//cv::Mat image = cv::imread("F:\\Foton\\ngnl_data\\training\\help\\anime\\" + std::to_string((D + epoch * DMAX) % 60000) + ".png", CV_8UC1);
			cv::Mat image = cv::imread("F:\\Foton\\ngnl_data\\training\\help\\" + std::to_string(rand() % 10) + "\\" + std::to_string((epoch * DMAX + D) % 6000 + 1) + ".png", CV_8UC1);

			cv::resize(image, image, cv::Size(image_size, image_size));
			BatchExp(image, out, 0);
			Exp(out, weight, discriminator.Layer, LayersNumD);

			outReal = out;
			float out1 = out[discriminator.Onum - discriminator.Layer[LayersNumD - 1]];

			Random(Gout, 0, 1, 0, generator.Layer[0], clock());
			Exp(Gout, Gweight, generator.Layer, LayersNumG);
			for (ull i = 0; i < generator.Layer[LayersNumG - 1]; i++) {
				out[i] = Gout[generator.Onum - generator.Layer[LayersNumG - 1] + i];
			}

			Exp(out, weight, discriminator.Layer, LayersNumD);

			float out2 = out[discriminator.Onum - discriminator.Layer[LayersNumD - 1]];


			float LossFucntionD = log(out1) + log(1 - out2);
			//std::cout << out1 << "   " << out2 << "   " << "LossD: " << LossFucntionD << std::endl;


			for (ull i = 0; i < discriminator.Layer[1]; i++) {
				float sumOld = 0;
				float sum = 0;
				ull OutNum = discriminator.Layer[0] + i;

				for (ull j = 0; j < discriminator.Layer[2]; j++) {
					ull WeightNum = discriminator.Layer[0] * discriminator.Layer[1] + i + j * discriminator.Layer[1];

					sumOld += (1 - out1) * weight[WeightNum];
					sum += out2 * weight[WeightNum];

					weight[WeightNum] = weight[WeightNum] + StepD * ((1 - out1) * outReal[OutNum] - out2 * out[OutNum]);
				}
				delOld[i] = sumOld * outReal[OutNum] * (1 - outReal[OutNum]);
				del[i] = sum * out[OutNum] * (1 - out[OutNum]);
			}

			for (ull i = 0; i < discriminator.Layer[0]; i++) {
				for (ull j = 0; j < discriminator.Layer[1]; j++) {
					ull OutNum = i;
					ull WeightNum = i + j * discriminator.Layer[0];

					weight[WeightNum] = weight[WeightNum] + StepD * (delOld[j] * outReal[i] - del[j] * out[i]);
				}
			}

			//Iter(out, outReal, weight, del, delOld, discriminator.Layer, LayersNumD);
		}

		for (ull G = 0; G < GMAX; G++) {
			Random(Gout, 0, 1, 0, generator.Layer[0], clock());
			Exp(Gout, Gweight, generator.Layer, LayersNumG);

			for (ull i = 0; i < generator.Layer[LayersNumG - 1]; i++) {
				out[i] = Gout[generator.Onum - generator.Layer[LayersNumG - 1] + i];
			}
			OutExp(out, 0);


			Exp(out, weight, discriminator.Layer, LayersNumD);
			float out2 = out[discriminator.Onum - 1];
			float LossFucntionG = log(out2);
			std::cout << out2 << "   " << "LossG: " << LossFucntionG << std::endl;

			ull WeightDopNum = 0, OutDopNum = 0, DelDopNum = 0;

			for (ull i = 0; i < discriminator.Layer[1]; i++) {
				float sumOld = 0;
				ull OutNum = discriminator.Layer[0] + i;

				for (ull j = 0; j < discriminator.Layer[2]; j++) {
					ull WeightNum = discriminator.Layer[0] * discriminator.Layer[1] + i + j * discriminator.Layer[1];

					sumOld += (1 - out2) * weight[WeightNum];
				}
				Gdel[i] = sumOld * out[OutNum] * (1 - out[OutNum]);
			}

			for (ull i = 0; i < discriminator.Layer[0]; i++) {
				float sum = 0;
				for (ull j = 0; j < discriminator.Layer[1]; j++) {
					ull OutNum = i;
					ull WeightNum = i + j * discriminator.Layer[0];
					sum += Gdel[j] * weight[WeightNum];
				}
				Gdel[discriminator.Layer[1] + i] = sum * out[i] * (1 - out[i]);
			}
			DelDopNum = discriminator.Layer[1];

			WeightDopNum = 0, OutDopNum = 0;


			for (ull i = 0; i < generator.Layer[1]; i++) {
				float per = 0;
				ull OutNum = generator.Layer[0] + i;

				for (ull j = 0; j < generator.Layer[2]; j++) {
					ull WeightNum = generator.Layer[0] * generator.Layer[1] + i + j * generator.Layer[1];
					per += Gdel[DelDopNum + j] * Gweight[WeightNum];

					Gweight[WeightNum] = Gweight[WeightNum] + StepG * Gdel[DelDopNum + j] * Gout[OutNum];
				}
				Gdel[DelDopNum + generator.Layer[2] + i] = per * (1 - Gout[OutNum]) * Gout[OutNum];
			}
			DelDopNum += generator.Layer[2];


			for (ull i = 0; i < generator.Layer[0]; i++) {
				for (ull j = 0; j < generator.Layer[1]; j++) {
					ull WeightNum = i + j * generator.Layer[0];

					Gweight[WeightNum] = Gweight[WeightNum] + StepG * Gdel[DelDopNum + j] * Gout[i];
				}
			}
		}
	}
}
