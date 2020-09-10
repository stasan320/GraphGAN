void Testing(cv::Mat image, double* out, double* weight, int coat, int* n) {
	int Onum = 0, Wnum = 0, Dnum = 0;
	for (int i = 0; i < coat; i++) {
		Onum = Onum + n[i];
		Wnum = Wnum + n[i] * n[i + 1];
	}

	std::cout << std::endl << "<                                                      > " << std::endl;
	//std::cout << "Testing" << std::endl;

	std::vector<int>DErrors(10, 0);
	for (int k = 0; k < n[coat - 1]; k++) {
		for (int test = 0; test < 890; test++) {
			image = cv::imread("D:\\Foton\\ngnl_data\\testing\\" + std::to_string(k) + "\\1 (" + std::to_string(test + 1) + ").png");
			if (!image.data) {
				std::cout << "Error upload image " << "D:\\Foton\\ngnl_data\\testing\\" + std::to_string(k) + "\\1 (" + std::to_string(test + 1) + ").png";
				exit(-1);
			}
			Image(image, out, 0);
			for (int i = 0; i < (coat - 1); i++) {
				SumFunc(out, weight, n, i);
				//std::cout << std::endl;
			}

			double data = 0;
			int number = 0;
			for (int i = 0; i < n[coat - 1]; i++) {
				if (data < out[Onum - n[coat - 1] + i]) {
					data = out[Onum - n[coat - 1] + i];
					number = i;
				}
			}
			if (k == number) {
				DErrors[k] = DErrors[k] + 1;
			}
		}
		std::cout << "                                                       >\r";
		std::cout << "< Result for " << k << ": " << std::setprecision(5) << (double)DErrors[k] / (double)890 << std::endl;

	}
	double data = 0;
	for (int i = 0; i < n[coat - 1]; i++) {
		data = DErrors[i] + data;
	}
	data = data / (890 * (double)n[coat - 1]);
	std::cout << "                                                       >\r";
	std::cout << "< Result for all: " << std::setprecision(5) << data << std::endl;
	std::cout << "<------------------------------------------------------>" << std::endl;
}

void GlobalTime(unsigned int TIME) {
	std::string sec = "", min = "", hour = "", ms = "";
	if (TIME % (1000) < 100) {
		ms = "0";
		if (TIME % (1000) < 10) {
			ms = "00";
		}
	}
	if (TIME / (1000) % (60) < 10) {
		sec = "0";
	}
	if (TIME / (1000 * 60) % (60) < 10) {
		min = "0";
	}
	if (TIME / (1000 * 60 * 24) % (24) < 10) {
		hour = "0";
	}

	std::cout << std::endl << std::endl << "<------------------------------------------------------>" << std::endl;
	std::cout << "                                                       >\r";
	std::cout << "< Global time: " << hour << TIME / (1000 * 24 * 60) % (24) << ":" << min << TIME / (1000 * 60) % (60) << ":" << sec << TIME / (1000) % (60) << "." << ms << TIME % 1000;

}
