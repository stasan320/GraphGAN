#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <torch/torch.h>
#include <ATen/ATen.h>

void Out(torch::Tensor tensor, int k, int image_size) {
	cv::Mat mat = cv::Mat(image_size, image_size, CV_32FC1, tensor.data_ptr());

	/*if (k % 10 == 0) {
		cv::Mat image(28, 28, CV_8UC1);
		for (int i = 0; i < 28; i++) {
			for (int j = 0; j < 28; j++) {
				image.at<uchar>(i, j) = ceil(mat.at<float>(i, j) * 255);
			}
		}
		cv::imwrite("D:\\Foton\\ngnl_data\\gen_image\\test\\" + std::to_string(time(NULL)) + ".png", image);
	}*/
	cv::imshow("Out", mat);
	cv::waitKey(1);

}

void ConsoleData(int epoch, int Epochs, int batch_index, int batches_per_epoch, float d_loss, float g_loss) {
	time_t t = time(NULL);
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
	std::cout << "[" << hour << ltm->tm_hour << ":" << min << ltm->tm_min << ":" << sec << ltm->tm_sec << "][" << epoch /*<< std::setprecision(5) */
		<< "/" << Epochs << "][" << batch_index << "/" << batches_per_epoch << "] Dis_loss: " << std::setprecision(4) << d_loss << " | Gen_loss: " << std::setprecision(4) << g_loss << "\r";

}
