/*void Random(double* Arr, double min, double max, int start, int end, unsigned long long MStime) {
	srand(MStime  + time(0));
	#include <random>
	std::random_device rd;
	std::mt19937 genr(rd());

	for (int i = start; i < end; i++) {
		Arr[i] = (double)(genr()) / rd.max() * (max - min) + min;
		//Arr[i] = (double)(rand() / RAND_MAX * (max - min)) + min;
		//Arr[i] = -0.5;
		//std::cout << Arr[i] << std::endl;
	}
}*/

void Random(double* Arr, double min, double max, int start, int end, unsigned long long MStime) {
	srand(static_cast<unsigned long long>(MStime + time(0)));
	for (int i = start; i < end; i++) {
		Arr[i] = (double)(rand()) / RAND_MAX * (max - min) + min;
	}
}

void SumFunc(double* out, double* weight, int* n, int num) {
	int Onum = 0, Wnum = 0;
	for (int i = 0; i < num; i++) {
		Onum = Onum + n[i];
		Wnum = Wnum + n[i] * n[i + 1];
	}

	for (int i = 0; i < n[num + 1]; i++) {
		double net = 0;
		for (int j = 0; j < n[num]; j++) {
			net = net + weight[Wnum + j + i * n[num]] * out[Onum + j];
			if (net != net) {
				std::cout << "sumfunc" << weight[Wnum + j + i * n[num]] << std::endl;
			}
		}
		out[Onum + n[num] + i] = 1.0 / (1.0 + exp(-net));
	}
}

void GenIterNull(double* out, double outO, double* weight, double* delw, double* del, int* n, int cout, int index, double dop) {
	int Onum = 0, Wnum = 0;

	for (int i = 0; i < (cout - 1); i++) {
		Onum = Onum + n[i];
	}

	del[index] = -log(outO) * dop;

	Onum = 0;

	for (int i = 0; i < (cout - 2); i++) {
		Onum = Onum + n[i];
		Wnum = Wnum + n[i] * n[i + 1];
	}

	for (int i = 0; i < n[cout - 2]; i++) {
		double grad = out[Onum + i] * del[index];
		delw[Wnum + i + index * n[cout - 2]] = 0.5 * grad + 0.03 * delw[Wnum + i + index * n[cout - 2]];
		weight[Wnum + i + index * n[cout - 2]] = weight[Wnum + i + index * n[cout - 2]] + delw[Wnum + i + index * n[cout - 2]];

	}
}

void DisIterNull(double* out, double* outO, double* weight, double* delw, double* del, int* n, int cout) {
	int Onum = 0, Wnum = 0, Dnum = 0;

	for (int i = 0; i < (cout - 1); i++) {
		Onum = Onum + n[i];
	}

	for (int i = 0; i < n[cout - 1]; i++) {
		del[i] = (outO[i] - out[Onum + i]) * (1 - out[Onum + i]) * out[Onum + i];
	}

	Onum = 0;

	for (int i = 0; i < (cout - 2); i++) {
		Onum = Onum + n[i];
		Wnum = Wnum + n[i] * n[i + 1];
	}

	for (int i = 0; i < n[cout - 2]; i++) {
		for (int j = 0; j < n[cout - 1]; j++) {
			double grad = out[Onum + i] * del[Dnum + j];
			delw[Wnum + i + j * n[cout - 2]] = 0.35 * grad + 0.03 * delw[Wnum + i + j * n[cout - 2]];
			weight[Wnum + i + j * n[cout - 2]] = weight[Wnum + i + j * n[cout - 2]] + delw[Wnum + i + j * n[cout - 2]];
			if (weight[Wnum + i + j * n[cout - 2]] != weight[Wnum + i + j * n[cout - 2]]) {
				std::cout << "Null iter" << weight[Wnum + i + j * n[cout - 2]] << std::endl;
			}
		}
	}
}

void Iter(double* out, double* weight, double* delw, double* del, int* n, int num, int coat) {
	int Onum = 0, Wnum = 0, Dnum = 0;
	for (int i = 0; i < (coat - num - 2); i++) {
		Onum = Onum + n[i];
		Wnum = Wnum + n[i] * n[i + 1];
	}

	for (int i = 0; i < (num); i++) {
		Dnum = Dnum + n[coat - i - 1];
	}

	for (int i = 0; i < n[coat - num - 2]; i++) {
		double per = 0;
		for (int j = 0; j < n[coat - num - 1]; j++) {
			per = per + del[Dnum + j] * weight[Wnum + i + n[coat - num - 2] * j];
		}
		del[Dnum + n[coat - num - 1] + i] = per * (1 - out[Onum + i]) * out[Onum + i];
	}
	Dnum = Dnum + n[coat - num - 1];
	Onum = 0, Wnum = 0;
	for (int i = 0; i < (coat - num - 3); i++) {
		Onum = Onum + n[i];
		Wnum = Wnum + n[i] * n[i + 1];
	}

	for (int i = 0; i < n[coat - num - 3]; i++) {
		for (int j = 0; j < n[coat - num - 2]; j++) {
			double grad = out[Onum + i] * del[Dnum + j];
			delw[Wnum + i + j * n[coat - num - 3]] = 0.35 * grad + 0.03 * delw[Wnum + i + j * n[coat - num - 3]];
			weight[Wnum + i + j * n[coat - num - 3]] = weight[Wnum + i + j * n[coat - num - 3]] + delw[Wnum + i + j * n[coat - num - 3]];
			if (weight[Wnum + i + j * n[coat - num - 3]] != weight[Wnum + i + j * n[coat - num - 3]]) {
				std::cout << "iter" << del[Dnum + j] << std::endl;
				exit(-1);
			}
		}
	}
}

void Image(cv::Mat image, double* out, int Onum) {
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			double per;
			per = image.at<cv::Vec3b>(i, j)[0];
			out[Onum + i * image.cols + j] = per / 255.0;
		}
	}

	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			out[Onum + i * image.cols + j] = 1.0;
		}
	}
}

void Out(cv::Mat image, double* out, int Onum) {
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			image.at<uchar>(i, j) = ceil(out[Onum + j + i * image.cols] * 255);
		}
	}
	cv::imshow("Out", image);
	cv::waitKey(1);
}

void ConfigPath(std::string &cpath) {
	std::string path;
	WCHAR buffer[MAX_PATH];
	GetModuleFileNameW(NULL, buffer, sizeof(buffer) / sizeof(buffer[0]));
	//printf(buffer);
	bool t = false;
	int i = 0;
	int conv;

	while (t == false) {
		if (buffer[wcslen(buffer) - 1 - i] != '\\') {
			buffer[wcslen(buffer) - 1 - i] = ' ';

		}
		else {
			t = true;
			conv = wcslen(buffer) - 1 - i;
		}
		i++;
	}
	for (int i = 0; i < conv; i++) {
		path += buffer[i];
		if (buffer[i] == '\\') {
			path += '\\';

		}
	}
	cpath = path;
}
