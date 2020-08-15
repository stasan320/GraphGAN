void Random(double* Arr, double min, double max, int start, int end, unsigned long long MStime) {
	srand(static_cast<unsigned long long>(MStime  + time(0)));
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
		}
		//out[Onum + n[num] + i] = (1 - exp(-2 * net)) / (1 + exp(-2 * net));;
		out[Onum + n[num] + i] = 1 / (1 + exp(-net));
		/*}
		else {
			out[Onum + n[num] + i] = 0;
		}*/
		//std::cout << out[Onum + n[num] + i] << std::endl;
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
			delw[Wnum + i + j * n[cout - 2]] = 0.7 * grad + 0.3 * delw[Wnum + i + j * n[cout - 2]];
			weight[Wnum + i + j * n[cout - 2]] = weight[Wnum + i + j * n[cout - 2]] + delw[Wnum + i + j * n[cout - 2]];
		}
	}
}

void Iter(double* out, double* weight, double* delw, double* del, int* n, int num, int coat) {
	int Onum = 0, Wnum = 0, Dnum = 0;
	for (int i = 0; i < (coat - num - 3); i++) {
		Onum = Onum + n[i];
		Wnum = Wnum + n[i] * n[i + 1];
	}

	for (int i = 0; i < (num + 1); i++) {
		Dnum = Dnum + n[coat - i - 1];
	}
	//std::cout << n[coat - num - 2] << std::endl;

	for (int i = 0; i < n[coat - num - 3]; i++) {
		double per = 0;
		for (int j = 0; j < n[coat - num - 2]; j++) {
			per = per + del[Dnum + j] * weight[Wnum + i + n[coat - num - 3] * j];
		}
		del[Dnum + n[coat - num - 2] + i] = per * (1 - out[Onum + i]) * out[Onum + i];
	}

	for (int i = 0; i < n[coat - num - 3]; i++) {
		for (int j = 0; j < n[coat - num - 2]; j++) {
			double grad = out[Onum + i] * del[Dnum + j];
			delw[Wnum + i + j * n[coat - num - 3]] = 0.5 * grad + 0.03 * delw[Wnum + i + j * n[coat - num - 3]];
			weight[Wnum + i + j * n[coat - num - 3]] = weight[Wnum + i + j * n[coat - num - 3]] + delw[Wnum + i + j * n[coat - num - 3]];
		}
	}
}

void Image(cv::Mat image, double* out, int Onum) {
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			double per;
			per = image.at<cv::Vec3b>(i, j)[0];
			per = per / 255 * 1 - 0;
			out[Onum + j + i * image.cols] = per;
			//std::cout << out[Onum + j + i * image.cols] << std::endl;
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
