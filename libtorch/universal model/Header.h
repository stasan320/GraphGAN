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



const int Batch = 60;

void Out(torch::Tensor tensor, int k, int image_size) {
	//image_size = 64;

	//for (int i = 0; i < Batch; i++) {
		cv::Mat mat = cv::Mat(image_size, image_size, CV_32FC1, tensor/*[i]*/.data_ptr());
		/*cv::Mat mats = cv::Mat(image_size, image_size, CV_32FC1, tensor[1].data_ptr());
		cv::Mat matss = cv::Mat(image_size, image_size, CV_32FC1, tensor[2].data_ptr());
		cv::Mat Out;



		std::vector<cv::Mat> channels = {mat, mats, matss};

		cv::merge(channels, Out);*/


		//cv::imwrite("D:\\Foton\\ngnl_data\\gen_image\\test\\" + std::to_string(time(NULL)) + ".png", mat);
		cv::imshow("Out", mat);
		cv::waitKey(1);
	//}
}
