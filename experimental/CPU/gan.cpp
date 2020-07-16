#include <windows.h>
#include <iostream>
#include <cmath>
#include <time.h>
#include "Header.h"

using namespace std;

const int coat = 3;


int main() {
	int  w = 0, n[coat] = { 2, 5, 1 }, kl, nc[coat] = { 2, 2, 2 };
	double iter = 1, per, Giter = 1;
	int min = -1, max = 1;

	double ka = 0;
	int var[2];

	for (int i = 0; i < coat; i++) {
		w = n[i] + w;
	}

	kl = w;

	std::cout << log(1e-32);
	//return 0;

	double* outO = new double[n[coat - 1]];
	double* del = new double[w - n[0]];
	double* out = new double[n[0] + n[1] + n[2]];
	w = 0;
	for (int i = 0; i < coat; i++) {
		w = nc[i] + w;
	}

	double* GoutO = new double[nc[coat - 1]];
	double* Gdel = new double[w - nc[0]];
	double* Gout = new double[nc[0] + nc[1] + nc[2]];
	double* Gweight = new double[nc[0] * nc[1] + nc[1] * nc[2]];
	double* Gdelw = new double[nc[0] * nc[1] + nc[1] * nc[2]];

	for (int i = 0; i < nc[0] * nc[1] + nc[1] * nc[2]; i++) {
		Gweight[i] = 1 / (1 + exp(-i));
		Gdelw[i] = 0;
	}

	for (int i = 0; i < (kl - n[0]); i++) {
		del[i] = 0;
	}

	for (int i = 0; i < n[0]; i++) {
		out[i] = 0 + i;
	}

	w = 0;

	for (int i = 0; i < (coat - 1); i++) {
		w = n[i] * n[i + 1] + w;
	}

	double* weight = new double[n[0] * n[1] + n[1] * n[2]];
	double* delw = new double[n[0] * n[1] + n[1] * n[2]];
	//double* grad = new double[w];

	for (int i = 0; i < n[0] * n[1] + n[1] * n[2]; i++) {
		weight[i] = 1 / (1 + exp(-i));
		delw[i] = 0;
	}


	GoutO[0] = 1;
	GoutO[1] = 1;
	var[0] = 1;
	var[1] = 1;


	/*outO[2] = 0.6;
	outO[3] = 0.785;
	outO[4] = 1;*/

	Random(Gout, nc[0]);

	for (int k = 0; k < 100000; k++) {
		for (int i = 0; i < 500; i++) {
			//Iter random data//
			//Random(Gout, nc[0]);
			Sum(Gweight, Gout, nc);
			for (int j = 0; j < n[0]; j++) {
				out[j] = Gout[nc[0] + nc[1] + j];
				//std::cout << out[j] << endl;
			}
			SumD(weight, out, n);
			//Out(out, n);
			//std::cout << std::endl;
			outO[0] = 0;
			DisIter(del, outO, out, weight, delw, n, iter);
			//iter = iter * 0.999;

			//}
			//Iter true data//
			for (int j = 0; j < 10; j++) {
				out[0] = 0.2;
				out[1] = 0.4;
				SumD(weight, out, n);
				//if ((out[n[0] + n[1]] > 0.8) && (k > 5)) {
				outO[0] = 1;
				DisIter(del, outO, out, weight, delw, n, iter);
				//iter = iter * 0.999;
			}
			//Out(out, n);
			//std::cout << std::endl;
		}
		//std::cout << k << endl;
		//return 0;

		for (int j = 0; j < 250; j++) {
			for (int i = 0; i < 2; i++) {
				//round 1
				//Random(Gout, nc[0]);
				Sum(Gweight, Gout, nc);
				//Out(Gout, nc);
				for (int j = 0; j < n[0]; j++) {
					out[j] = Gout[nc[0] + nc[1] + j];
					std::cout << Gout[nc[0] + nc[1] + j] << std::endl;
				}
				//std::cout << endl;
				Sum(weight, out, n);
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
				//del[i] = (1 - GoutO[0]);
				per = 1;
				GenIter(Gdel, GoutO, Gout, Gweight, Gdelw, nc, var, i, per, Giter);
				Giter = Giter * 0.999;

				//round 2
				//Random(Gout, nc[0]);
				Sum(Gweight, Gout, nc);
				for (int j = 0; j < n[0]; j++) {
					out[j] = Gout[nc[0] + nc[1] + j];
				}
				Sum(weight, out, n);
				for (int j = 0; j < n[2]; j++) {
					GoutO[1] = GoutO[0];
					GoutO[0] = out[n[0] + n[1] + j];
					//std::cout << GoutO[0] - GoutO[1] << std::endl;
				}
				if ((GoutO[0] - GoutO[1]) < 0) {
					//del[i] = (1 - GoutO[0]);
					per = -1.7;
					GenIter(Gdel, GoutO, Gout, Gweight, Gdelw, nc, var, i, per, Giter);
				}
			}
		}
	}
	return 0;
}
