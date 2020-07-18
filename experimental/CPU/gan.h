void SumG(float* weight, float* out, int* n, int layer) {
	int Onum = 0, Wnum = 0;
	for (int k = 0; k < (layer - 1); k++) {
		for (int i = 0; i < n[k + 1]; i++) {
			float net = 0;
			for (int j = 0; j < n[k]; j++) {
				net = weight[Wnum + i * n[k] + j] * out[Onum + j] + net;
			}
			out[Onum + n[k] + i] = 1 / (1 + exp(-net));
		}
		Wnum = Wnum + n[k] * n[k + 1];
		Onum = Onum + n[k];
	}
}

void SumD(float* weight, float* out, int* n, int layer) {
	int Onum = 0, Wnum = 0;
	for (int k = 0; k < (layer - 1); k++) {
		for (int i = 0; i < n[k + 1]; i++) {
			float net = 0;
			for (int j = 0; j < n[k]; j++) {
				net = weight[Wnum + i * n[k] + j] * out[Onum + j] + net;
			}
			out[Onum + n[k] + i] = 1 / (1 + exp(-net));
		}
		Wnum = Wnum + n[k] * n[k + 1];
		Onum = Onum + n[k];
	}

}

void DisIter(float* del, float* outO, float* out, float* weight, float* delw, int* n, float iter, int layer, int NeuralSum, int WeightSum) {
	int Dnum = 0, Onum = NeuralSum - n[layer - 1], Wnum = WeightSum;

	for (int i = 0; i < n[2]; i++) {
		del[i] = (outO[i] - out[n[0] + n[1] + i]) * out[n[0] + n[1] + i] * (1 - out[n[0] + n[1] + i]) * iter;
	}

	for (int k = 1; k < (layer - 1); k++) {
		Onum = Onum - n[layer - k - 1];
		Wnum = Wnum - n[layer - k - 1] * n[layer - k];
		for (int i = 0; i < n[layer - k - 1]; i++) {
			float per = 0;
			for (int j = 0; j < n[layer - k]; j++) {
				per = per + del[Dnum + j] * weight[Wnum + i + n[layer - k - 1] * j];
			}
			del[Dnum + n[layer - k] + i] = (1 - out[Onum + i]) * out[Onum + i] * per;
		}
		Dnum = Dnum + n[layer - k];
	}

	Dnum = 0;
	Onum = n[0];
	Wnum = n[0] * n[1];

	for (int i = 0; i < n[1]; i++) {
		float grad = 0;
		for (int j = 0; j < n[2]; j++) {
			grad = del[Dnum + j] * out[Onum + i];
			delw[Wnum + i + n[layer - 2] * j] = 0.5 * grad + 0.3 * delw[Wnum + i + n[layer - 2] * j];
			weight[Wnum + i + n[layer - 2] * j] = weight[Wnum + i + n[layer - 2] * j] + delw[Wnum + i + n[layer - 2] * j];
		}
	}

	Dnum = n[2];
	Onum = 0;
	Wnum = 0;

	for (int i = 0; i < n[0]; i++) {
		float grad = 0;
		for (int j = 0; j < n[1]; j++) {
			grad = del[Dnum + j] * out[Onum + i];
			delw[Wnum + i + n[layer - 3] * j] = 0.5 * grad + 0.3 * delw[Wnum + i + n[layer - 3] * j];
			weight[Wnum + i + n[layer - 3] * j] = weight[Wnum + i + n[layer - 3] * j] + delw[Wnum + i + n[layer - 3] * j];
		}
	}
}

void GenIter(float* del, float* outO, float* out, float* weight, float* delw, int* n, int* var, int ind, float k, float iter) {
	int layer = 3;
	int Dnum = 0, Onum = 0, Wnum = 0;

	del[ind] = -log(outO[0]) * k * iter * (1 - out[n[0] + n[1] + ind]) * out[n[0] + n[1] + ind];

	Dnum = n[2];
	Wnum = n[0] * n[1];

	for (int i = 0; i < n[layer - 2]; i++) {
		float per = 0;
		per = per + del[ind] * weight[Wnum + i + n[layer - 2] * ind];
		del[Dnum + i] = (1 - out[n[0] + i]) * out[n[0] + i] * per;
	}

	Dnum = 0;
	Onum = n[0];
	Wnum = n[0] * n[1];

	for (int i = 0; i < n[1]; i++) {
		float grad = 0;
		grad = del[Dnum + ind] * out[Onum + i];
		//std::cout << grad << std::endl;
		delw[Wnum + i + n[1] * ind] = 0.5 * grad + 0.3 * delw[Wnum + i + n[1] * ind];
		weight[Wnum + i + n[1] * ind] = weight[Wnum + i + n[1] * ind] + delw[Wnum + i + n[1] * ind];
	}

	Dnum = n[2];
	Onum = 0;
	Wnum = 0;

	for (int i = 0; i < n[0]; i++) {
		float grad = 0;
		for (int j = 0; j < n[1]; j++) {
			grad = del[Dnum + j] * out[Onum + i];
			delw[Wnum + i + n[layer - 3] * j] = 0.5 * grad + 0.3 * delw[Wnum + i + n[layer - 3] * j];
			weight[Wnum + i + n[layer - 3] * j] = weight[Wnum + i + n[layer - 3] * j] + delw[Wnum + i + n[layer - 3] * j];
		}
	}
}


void Random(float* out, int n) {
	srand(static_cast<unsigned int>(clock()));
	for (int i = 0; i < n; i++) {
		float per;
		per = (float)(rand()) / RAND_MAX * 2 - 1;
		out[i] = (float)(rand()) / RAND_MAX;
	}
}

void Out(float* out, int* n) {
	for (int i = 0; i < n[2]; i++) {
		std::cout << out[n[0] + n[1] + i] << std::endl;
	}
}
