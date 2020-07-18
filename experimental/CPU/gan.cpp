#include <windows.h>
#include <iostream>
#include <cmath>
#include <time.h>
#include "Header.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

using namespace std;

const int layer = 3;


int main() {
	int  w = 0, n[layer] = { 2, 2, 1 }, kl, nc[layer] = { 2, 3, 2 };
	float iter = 1, per, Giter = 1;
	int min = -1, max = 1, DNeuralSum = 0, GNeuralSum = 0, DWeightSum = 0, GWeightSum = 0;

	for (int i = 0; i < layer; i++) {
		DNeuralSum = n[i] + DNeuralSum;
	}
	for (int i = 0; i < layer - 1; i++) {
		DWeightSum = n[i] * n[i + 1] + DWeightSum;
	}

	for (int i = 0; i < layer; i++) {
		GNeuralSum = nc[i] + GNeuralSum;
	}
	for (int i = 0; i < layer - 1; i++) {
		GWeightSum = nc[i] * nc[i + 1] + GWeightSum;
	}

	float* outO = new float[n[layer - 1]];
	float* del = new float[DNeuralSum - n[0]];
	float* out = new float[DNeuralSum];
	float* weight = new float[DWeightSum];
	float* delw = new float[DWeightSum];

	float* GoutO = new float[nc[layer - 1]];
	float* Gdel = new float[GNeuralSum - nc[0]];
	float* Gout = new float[GNeuralSum];
	float* Gweight = new float[GWeightSum];
	float* Gdelw = new float[GWeightSum];

	for (int i = 0; i < GWeightSum; i++) {
		Gweight[i] = 1 / (1 + exp(-i));
		Gdelw[i] = 0;
	}

	for (int i = 0; i < DWeightSum; i++) {
		weight[i] = 1 / (1 + exp(-i));
		delw[i] = 0;
	}


	GoutO[0] = 1;
	GoutO[1] = 1;

	Random(Gout, nc[0]);

	for (int k = 0; k < 100000; k++) {
		for (int i = 0; i < 450; i++) {
			//Iter random data//
			//Random(Gout, nc[0]);
			SumG(Gweight, Gout, nc, layer);
			for (int j = 0; j < n[0]; j++) {
				out[j] = Gout[nc[0] + nc[1] + j];
				//std::cout << out[j] << endl;
			}
			SumD(weight, out, n, layer);
			//Out(out, n);
			//std::cout << std::endl;
			outO[0] = 0;
			DisIter(del, outO, out, weight, delw, n, iter, layer, DNeuralSum, DWeightSum);
			//iter = iter * 0.999;

			//}
			//Iter true data//
			for (int j = 0; j < 10; j++) {
				out[0] = 0.2;
				out[1] = 0.5;
				SumD(weight, out, n, layer);
				//if ((out[n[0] + n[1]] > 0.8) && (k > 5)) {
				outO[0] = 1;
				DisIter(del, outO, out, weight, delw, n, iter, layer, DNeuralSum, DWeightSum);
				//iter = iter * 0.999;
			}
			//Out(out, n);
			//std::cout << std::endl;
		}

		for (int j = 0; j < 250; j++) {
			for (int i = 0; i < 2; i++) {
				//round 1
				//Random(Gout, nc[0]);
				SumG(Gweight, Gout, nc, layer);
				//Out(Gout, nc);
				for (int j = 0; j < n[0]; j++) {
					out[j] = Gout[nc[0] + nc[1] + j];
					std::cout << Gout[nc[0] + nc[1] + j] << std::endl;
				}
				//std::cout << endl;
				SumD(weight, out, n, layer);
				for (int j = 0; j < n[2]; j++) {
					//std::cout << "Error " << out[n[0] + n[1] + j] << endl;
				}
				//Out(out, n);
				std::cout << std::endl;
				for (int j = 0; j < n[2]; j++) {
					GoutO[1] = GoutO[0];
					GoutO[0] = out[n[0] + n[1] + j];
					//std::cout << GoutO[0] - GoutO[1] << std::endl;
				}
				per = 1;
				GenIter(Gdel, GoutO, Gout, Gweight, Gdelw, nc, i, per, Giter);
				Giter = Giter * 0.999;

				//round 2
				//Random(Gout, nc[0]);
				SumG(Gweight, Gout, nc, layer);
				for (int j = 0; j < n[0]; j++) {
					out[j] = Gout[nc[0] + nc[1] + j];
				}
				SumD(weight, out, n, layer);
				for (int j = 0; j < n[2]; j++) {
					GoutO[1] = GoutO[0];
					GoutO[0] = out[n[0] + n[1] + j];
					//std::cout << GoutO[0] - GoutO[1] << std::endl;
				}
				if ((GoutO[0] - GoutO[1]) <= 0) {
					per = -1.5;
					GenIter(Gdel, GoutO, Gout, Gweight, Gdelw, nc, i, per, Giter);
				}
			}
			//Giter = Giter * 0.999;
		}
	}
	return 0;
}
