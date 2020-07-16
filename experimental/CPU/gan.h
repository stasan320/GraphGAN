void Sum(double* weight, double* out, int* n) {
	for (int i = 0; i < n[1]; i++) {
		double net = 0;
		for (int j = 0; j < n[0]; j++) {
			net = weight[i * n[0] + j] * out[j] + net;
		}
		//out[n[0] + i] = 1 / (1 + exp(-net));
		/*if (net < 0) {
			out[n[0] + i] = 0;
		}
		else {
			out[n[0] + i] = net;
		}*/
		out[n[0] + i] = 1 / (1 + exp(-net));
	}

	for (int i = 0; i < n[2]; i++) {
		double net = 0;
		for (int j = 0; j < n[1]; j++) {
			net = weight[n[0] * n[1] + i * n[1] + j] * out[n[0] + j] + net;
		}
		//out[n[0] + n[1] + i] = (exp(2 * net) - 1) / (1 + exp(2 * net));
		out[n[0] + n[1] + i] = 1 / (1 + exp(-net));
		/*if (net < 0) {
			out[n[0] + n[1] + i] = 0;
		}
		else {
			out[n[0] + n[1] + i] = net;
		}*/
	}
}

void SumD(double* weight, double* out, int* n) {
	for (int i = 0; i < n[1]; i++) {
		double net = 0;
		for (int j = 0; j < n[0]; j++) {
			net = weight[i * n[0] + j] * out[j] + net;
		}
		out[n[0] + i] = 1 / (1 + exp(-net));
	}

	for (int i = 0; i < n[2]; i++) {
		double net = 0;
		for (int j = 0; j < n[1]; j++) {
			net = weight[n[0] * n[1] + i * n[1] + j] * out[n[0] + j] + net;
		}
		out[n[0] + n[1] + i] = 1 / (1 + exp(-net));
	}
}

void DisIter(double* del, double* outO, double* out, double* weight, double* delw, int* n, double iter) {
	int coat = 3;
	int Dnum = 0, Onum = 0, Wnum = 0;

	for (int i = 0; i < n[2]; i++) {
		del[i] = (outO[i] - out[n[0] + n[1] + i]) * out[n[0] + n[1] + i] * (1 - out[n[0] + n[1] + i]) * iter;
	}
	Dnum = n[2];
	Wnum = n[0] * n[1];

	for (int i = 0; i < n[coat - 2]; i++) {
		double per = 0;
		for (int j = 0; j < n[coat - 1]; j++) {
			per = per + del[j] * weight[Wnum + i + n[coat - 2] * j];
		}
		del[Dnum + i] = (1 - out[n[0] + i]) * out[n[0] + i] * per;
	}

	Dnum = 0;
	Onum = n[0];
	Wnum = n[0] * n[1];

	for (int i = 0; i < n[1]; i++) {
		double grad = 0;
		for (int j = 0; j < n[2]; j++) {
			grad = del[Dnum + j] * out[Onum + i];
			delw[Wnum + i + n[coat - 2] * j] = 0.5 * grad + 0.3 * delw[Wnum + i + n[coat - 2] * j];
			weight[Wnum + i + n[coat - 2] * j] = weight[Wnum + i + n[coat - 2] * j] + delw[Wnum + i + n[coat - 2] * j];
		}
	}

	Dnum = n[2];
	Onum = 0;
	Wnum = 0;

	for (int i = 0; i < n[0]; i++) {
		double grad = 0;
		for (int j = 0; j < n[1]; j++) {
			grad = del[Dnum + j] * out[Onum + i];
			delw[Wnum + i + n[coat - 3] * j] = 0.5 * grad + 0.3 * delw[Wnum + i + n[coat - 3] * j];
			weight[Wnum + i + n[coat - 3] * j] = weight[Wnum + i + n[coat - 3] * j] + delw[Wnum + i + n[coat - 3] * j];
		}
	}
}

void GenIter(double* del, double* outO, double* out, double* weight, double* delw, int* n, int* var, int ind, double k, double iter) {
	int coat = 3;
	int Dnum = 0, Onum = 0, Wnum = 0;

	del[ind] = -log(outO[0]) * k * iter /** (1 - out[n[0] + n[1] + ind]) * out[n[0] + n[1] + ind]*/;

	Dnum = n[2];
	Wnum = n[0] * n[1];

	for (int i = 0; i < n[coat - 2]; i++) {
		double per = 0;
		per = per + del[ind] * weight[Wnum + i + n[coat - 2] * ind];
		del[Dnum + i] = (1 - out[n[0] + i]) * out[n[0] + i] * per;
	}

	Dnum = 0;
	Onum = n[0];
	Wnum = n[0] * n[1];

	for (int i = 0; i < n[1]; i++) {
		double grad = 0;
		grad = del[Dnum + ind] * out[Onum + i];
		//std::cout << grad << std::endl;
		delw[Wnum + i + n[1] * ind] = 0.5 * grad + 0.3 * delw[Wnum + i + n[1] * ind];
		weight[Wnum + i + n[1] * ind] = weight[Wnum + i + n[1] * ind] + delw[Wnum + i + n[1] * ind];
	}

	Dnum = n[2];
	Onum = 0;
	Wnum = 0;

	for (int i = 0; i < n[0]; i++) {
		double grad = 0;
		for (int j = 0; j < n[1]; j++) {
			grad = del[Dnum + j] * out[Onum + i];
			delw[Wnum + i + n[coat - 3] * j] = 0.5 * grad + 0.3 * delw[Wnum + i + n[coat - 3] * j];
			weight[Wnum + i + n[coat - 3] * j] = weight[Wnum + i + n[coat - 3] * j] + delw[Wnum + i + n[coat - 3] * j];
		}
	}
}


void Random(double* out, int n) {
	srand(static_cast<unsigned int>(clock()));
	for (int i = 0; i < n; i++) {
		double per;
		per = (double)(rand()) / RAND_MAX * 2 - 1;
		out[i] = (double)(rand()) / RAND_MAX;
		//cout << out[i] << endl;
	}
}

void Out(double* out, int* n) {
	for (int i = 0; i < n[2]; i++) {
		std::cout << out[n[0] + n[1] + i] << std::endl;
	}
}
