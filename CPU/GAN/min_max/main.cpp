#include "Header.h"

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
	std::vector<float> GdelOld(discriminator.Onum, 0);

	Random(weight, -1, 1, 0, discriminator.Wnum, clock());
	Random(Gweight, -1, 1, 0, generator.Wnum, clock());

	int dop = 0;

	//Random(Gout, 0, 1, 0, generator.Layer[0], clock());


	for (ull epoch = 0; epoch < 10000; epoch++) {
		for (ull D = 0; D < D_max; D++) {
			cv::Mat image = cv::imread("G:\\Foton\\data\\cropped\\1 (" + std::to_string((epoch * D_max + D + dop) % 5000 + 1) + ").jpg", CV_8UC1);
			if (!image.data) {
				dop++;
				image = cv::imread("G:\\Foton\\data\\cropped\\1 (" + std::to_string((epoch * D_max + D + dop) % 5000 + 1) + ").jpg", CV_8UC1);
			}
			//cv::Mat image = cv::imread("C:\\Users\\stasan\\Downloads\\archive\\mnist_png\\testing\\" + std::to_string(rand() % 10) + "\\1 (" + std::to_string((epoch * D_max + D) % 890 + 1) + ").png", CV_8UC1);


			cv::resize(image, image, cv::Size(image_size, image_size));
			BatchExp(image, out, 0);
			Exp(out, weight, discriminator.Layer, LayersNumD);

			outReal = out;
			float out1 = out[discriminator.Onum - discriminator.Layer[LayersNumD - 1]];

			Random(Gout, -0.5, 0, 0, generator.Layer[0], clock());
			Exp(Gout, Gweight, generator.Layer, LayersNumG);
			for (ull i = 0; i < generator.Layer[LayersNumG - 1]; i++) {
				out[i] = Gout[generator.Onum - generator.Layer[LayersNumG - 1] + i];
			}

			//Out(out, 0);



			Exp(out, weight, discriminator.Layer, LayersNumD);
			float out2 = out[discriminator.Onum - discriminator.Layer[LayersNumD - 1]];


			float LossFucntionD = log(out1) + log(1 - out2);
			std::cout << out1 << "   " << out2 << "   " << "LossD: " << LossFucntionD << std::endl;


			for (ull i = 0; i < discriminator.Layer[LayersNumD - 2]; i++) {
				for (ull j = 0; j < discriminator.Layer[LayersNumD - 1]; j++) {
					ull OutNum = discriminator.Onum - discriminator.Layer[LayersNumD - 1] - discriminator.Layer[LayersNumD - 2];
					ull WeightNum = discriminator.Wnum - discriminator.Layer[LayersNumD - 1] * discriminator.Layer[LayersNumD - 2] + j * discriminator.Layer[LayersNumD - 2] + i;


					delOld[i] = (1 - out1) * outReal[OutNum + i];
					del[i] = out2 * out[OutNum + i];
					weight[WeightNum] = weight[WeightNum] + StepD * (delOld[i] - del[i]);
				}
			}

			Iter(out, outReal, weight, del, delOld, discriminator.Layer, LayersNumD);
		}


		for (ull G = 0; G < G_max; G++) {
			Random(Gout, -0.5, 0, 0, generator.Layer[0], clock());
			Exp(Gout, Gweight, generator.Layer, LayersNumG);

			for (ull i = 0; i < generator.Layer[LayersNumG - 1]; i++) {
				out[i] = Gout[generator.Onum - generator.Layer[LayersNumG - 1] + i];
			}
			Out(out, 0);


			Exp(out, weight, discriminator.Layer, LayersNumD);
			float out2 = out[discriminator.Onum - 1];
			float LossFucntionG = log(out2);
			std::cout << out2 << "   " << "LossG: " << LossFucntionG << std::endl;

			ull WeightDopNum = 0, OutDopNum = 0, DelDopNum = 0;
			WeightDopNum += discriminator.Layer[LayersNumD - 2];
			OutDopNum += discriminator.Layer[LayersNumD - 1] + discriminator.Layer[LayersNumD - 2];

			for (ull i = 0; i < discriminator.Layer[LayersNumD - 2]; i++) {
				ull WeightNum = discriminator.Wnum - WeightDopNum + i;
				ull OutNum = discriminator.Onum - OutDopNum + i;

				Gdel[i] = (1 - out2) * weight[WeightNum] * out[OutNum] * (1 - out[OutNum]);
			}

			/*if (LayersNumD == 3) {
				for (ull i = 0; i < discriminator.Layer[0]; i++) {
					float Sum = 0;
					for (ull j = 0; j < discriminator.Layer[1]; j++) {
						ull WeightNum = i + j * discriminator.Layer[0];
						Sum += Gdel[j] * weight[WeightNum];
					}
					Gdel[discriminator.Layer[1] + i] = Sum * out[i] * (1 - out[i]);
				}

				DelDopNum += discriminator.Layer[1];
			}*/

			for (ull num = 0; num < (LayersNumD - 2); num++) {
				WeightDopNum += discriminator.Layer[LayersNumD - 2 - num] * discriminator.Layer[LayersNumD - 3 - num];
				OutDopNum += discriminator.Layer[LayersNumD - 2 - num];

				for (ull i = 0; i < discriminator.Layer[LayersNumD - 3 - num]; i++) {
					float sum = 0;
					for (ull j = 0; j < discriminator.Layer[LayersNumD - 2 - num]; j++) {
						ull WeightNum = discriminator.Wnum - WeightDopNum + i + j * discriminator.Layer[LayersNumD - 3 - num];
						//int OutNum = discriminator.Onum - OutDopNum + j;

						sum += (Gdel[DelDopNum + j] * weight[WeightNum])/* / (float)discriminator.Layer[LayersNumD - 2 - num]*/;
					}
					ull OutNum = discriminator.Onum - OutDopNum - discriminator.Layer[LayersNumD - 3 - num] + i;
					Gdel[DelDopNum + discriminator.Layer[LayersNumD - 2 - num] + i] = sum * out[OutNum] * (1 - out[OutNum]);
				}
				DelDopNum += discriminator.Layer[LayersNumD - 2 - num];
			}
			WeightDopNum = 0, OutDopNum = 0;

			for (ull num = 0; num < (LayersNumG - 2); num++) {
				WeightDopNum += generator.Layer[LayersNumG - 2 - num] * generator.Layer[LayersNumG - 1 - num];
				OutDopNum += generator.Layer[LayersNumG - 2 - num];

				for (ull i = 0; i < generator.Layer[LayersNumG - 2 - num]; i++) {
					float per = 0;
					ull OutNum = generator.Onum - OutDopNum + i;

					for (ull j = 0; j < generator.Layer[LayersNumG - 1 - num]; j++) {
						ull WeightNum = generator.Wnum - WeightDopNum + i + j * generator.Layer[LayersNumG - 2 - num];
						Gweight[WeightNum] = Gweight[WeightNum] + Step * Gdel[DelDopNum + j] * Gout[OutNum];

						per += Gdel[DelDopNum + j] * Gweight[WeightNum];
					}

					Gdel[DelDopNum + generator.Layer[LayersNumG - 1 - num] + i] = per * (1 - Gout[OutNum]) * Gout[OutNum];
				}
				DelDopNum += generator.Layer[LayersNumG - 1 - num];
			}


			for (ull i = 0; i < generator.Layer[0]; i++) {
				for (ull j = 0; j < generator.Layer[1]; j++) {
					ull WeightNum = i + j * generator.Layer[0];

					Gweight[WeightNum] = Gweight[WeightNum] + Step * Gdel[DelDopNum + j] * Gout[i];
				}
			}
		}
	}
}
