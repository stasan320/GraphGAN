#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>


#include <sstream>
#include <fstream>
#include <vector>
#include <tuple>



#include <iostream>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <torch/script.h>





void Out(torch::Tensor tensor, int k, int image_size) {
	//image_size = 64;
	cv::Mat mat = cv::Mat(image_size, image_size, CV_32FC1, tensor[0].data_ptr());
	cv::Mat mats = cv::Mat(image_size, image_size, CV_32FC1, tensor[1].data_ptr());
	cv::Mat matss = cv::Mat(image_size, image_size, CV_32FC1, tensor[2].data_ptr());
	cv::Mat Out;



	std::vector<cv::Mat> channels = {mat, mats, matss};

	cv::merge(channels, Out);


	//cv::imwrite("D:\\Foton\\ngnl_data\\gen_image\\test\\" + std::to_string(time(NULL)) + ".png", image);
	cv::imshow("Out", Out);
	cv::waitKey(100);
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
		<< "/" << Epochs << "][" << batch_index << "/" << batches_per_epoch << "] Dis_loss: "<< std::setprecision(4) << d_loss << " | Gen_loss: " << std::setprecision(4) << g_loss << "\r";

}
