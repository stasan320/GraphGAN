void Random(float* Arr, float min, float max, int start, int end, unsigned long int time) {
	srand(static_cast<unsigned int>(time));
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

void IterNull(float* out, float* outO, float* weight, float* delw, float* del, int* n, int cout) {
	int Onum = 0, Wnum = 0;

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
			float grad = out[Onum + i] * del[j];
			delw[Wnum + i + j * n[cout - 2]] = 0.5 * grad + 0.03 * delw[Wnum + i + j * n[cout - 2]];
			weight[Wnum + i + j * n[cout - 2]] = weight[Wnum + i + j * n[cout - 2]] + delw[Wnum + i + j * n[cout - 2]];
		}
	}
}

void Iter(float* out, float* weight, float* delw, float* del, int* n, int num, int coat) {
	int Onum = 0, Wnum = 0, Dnum = 0;
	for (int i = 0; i < (num - 1); i++) {
		Onum = Onum + n[i];
		Wnum = Wnum + n[i] * n[i + 1];
	}

	for (int i = (num + 1); i < coat; i++) {
		Dnum = Dnum + n[i];
	}

	for (int i = 0; i < n[num - 1]; i++) {
		float per = 0;
		for (int j = 0; j < n[num]; j++) {
			per = per + del[Dnum + i] * weight[Wnum + i + n[num - 1] * j];
		}
		del[Dnum + n[num] + i] = (1 - out[Onum + i]) * out[Onum + i] * per;
	}

	for (int i = 0; i < n[num - 1]; i++) {
		for (int j = 0; j < n[num]; j++) {
			float grad = out[Onum + i] * del[j];
			delw[Wnum + i + j * n[num - 2]] = 0.5 * grad + 0.03 * delw[Wnum + i + j * n[num - 2]];
			weight[Wnum + i + j * n[num - 2]] = weight[Wnum + i + j * n[num - 2]] + delw[Wnum + i + j * n[num - 2]];
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

void Out(cv::Mat result, float* out, int Onum) {
	for (int i = 0; i < result.rows; i++) {
		for (int j = 0; j < result.cols; j++) {
			result.at<uchar>(i, j) = ceil(out[Onum + j + i * result.cols] * 255);
		}
	}
	cv::imshow("Out", result);
	cv::waitKey(1);
}
