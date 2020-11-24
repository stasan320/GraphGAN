void Time(clock_t t, int epoch, float error, int* n, int coat) {
		t = time(NULL);
		std::string sec = "", min = "", hour = "";

		time_t now = time(NULL);
		tm* ltm = localtime(&now);
		std::cout << "                                                  \r";
		if (ltm->tm_sec < 10) {
			sec = "0";
		}
		if (ltm->tm_min < 10) {
			min = "0";
		}
		if (ltm->tm_hour < 10) {
			hour = "0";
		}
		std::cout << "[" << hour << ltm->tm_hour << ":" << min << ltm->tm_min << ":" << sec << ltm->tm_sec << "][" << epoch << "][" << std::setprecision(5) << error / n[coat - 1] << "] \r";

}
