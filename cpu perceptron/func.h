void Random(float* Arr, float min, float max, int start, int end, unsigned long long time) {
	srand(static_cast<unsigned long long>(time));
	for (int i = start; i < end; i++) {
		Arr[i] = (float)(rand()) / RAND_MAX * (max - min) + min;
	}
}

void SumFunc(float* out, float* weight, int* n, int num) {
	int Onum = 0, Wnum = 0;
	for (int i = 0; i < num; i++) {
		Onum = Onum + n[i];
		Wnum = Wnum + n[i] * n[i + 1];
	}

	for (int i = 0; i < n[num + 1]; i++) {
		float net = 0;
		for (int j = 0; j < n[num]; j++) {
			net = net + weight[Wnum + i * n[num] + j] * out[Onum + j];
		}
		out[Onum + n[num] + i] = 1 / (1 + exp(-net));
	}
}

void GenIterNull(float* out, float outO, float* weight, float* delw, float* del, int* n, int cout, int index, float dop) {
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
		float grad = out[Onum + i] * del[index];
		delw[Wnum + i + index * n[cout - 2]] = 0.5 * grad + 0.03 * delw[Wnum + i + index * n[cout - 2]];
		weight[Wnum + i + index * n[cout - 2]] = weight[Wnum + i + index * n[cout - 2]] + delw[Wnum + i + index * n[cout - 2]];

	}
}

void DisIterNull(float* out, float* outO, float* weight, float* delw, float* del, int* n, int cout) {
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
	//std::cout << Wnum << std::endl;

	for (int i = 0; i < n[cout - 2]; i++) {
		for (int j = 0; j < n[cout - 1]; j++) {
			float grad = out[Onum + i] * del[Dnum + j];
			delw[Wnum + i + j * n[cout - 2]] = 0.5 * grad + 0.03 * delw[Wnum + i + j * n[cout - 2]];
			weight[Wnum + i + j * n[cout - 2]] = weight[Wnum + i + j * n[cout - 2]] + delw[Wnum + i + j * n[cout - 2]];
		}
	}
}

void Iter(float* out, float* weight, float* delw, float* del, int* n, int num, int coat) {
	int Onum = 0, Wnum = 0, Dnum = 0;
	for (int i = 0; i < (coat - num - 3); i++) {
		Onum = Onum + n[i];
		Wnum = Wnum + n[i] * n[i + 1];
	}

	for (int i = (coat - 1); i > (coat - num - 2); i--) {
		Dnum = Dnum + n[i];
	}
	//std::cout << Dnum << std::endl;

	for (int i = 0; i < n[coat - num - 3]; i++) {
		float per = 0;
		for (int j = 0; j < n[coat - num - 2]; j++) {
			per = per + del[Dnum + i] * weight[Wnum + i + n[coat - num - 2] * j];
		}
		del[Dnum + n[coat - num - 2] + i] = (1 - out[Onum + i]) * out[Onum + i] * per;
	}

	for (int i = 0; i < n[coat - num - 3]; i++) {
		for (int j = 0; j < n[coat - num - 2]; j++) {
			float grad = out[Onum + i] * del[Dnum + j];
			delw[Wnum + i + j * n[coat - num - 3]] = 0.5 * grad + 0.03 * delw[Wnum + i + j * n[coat - num - 3]];
			weight[Wnum + i + j * n[coat - num - 3]] = weight[Wnum + i + j * n[coat - num - 3]] + delw[Wnum + i + j * n[coat - num - 3]];
		}
	}
}

void Image(cv::Mat image, float* out, int Onum) {
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			float per;
			per = image.at<cv::Vec3b>(i, j)[0];
			per = per / 255;
			out[Onum + j + i * image.cols] = per;
		}
	}
}

void Out(cv::Mat image, float* out, int Onum) {
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			image.at<uchar>(i, j) = ceil(out[Onum + j + i * image.cols] * 255);
		}
	}
	cv::imshow("Out", image);
	cv::waitKey(1);
}
