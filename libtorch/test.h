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
    cv::Mat mat = cv::Mat(image_size, image_size, CV_32FC1, tensor.data_ptr());
    cv::Mat mats = cv::Mat(28, 28, CV_32FC1, tensor.data_ptr());

    /*if (k % 10 == 0) {
        cv::Mat image(mat.rows, mat.cols, CV_8UC1);
        for (int i = 0; i < mat.rows; i++) {
            for (int j = 0; j < mat.cols; j++) {
                image.at<uchar>(i, j) = ceil(mat.at<float>(i, j) * 255);
            }
        }

        cv::Mat images(mats.rows, mats.cols, CV_8UC1);
        for (int i = 0; i < mats.rows; i++) {
            for (int j = 0; j < mats.cols; j++) {
                images.at<uchar>(i, j) = ceil(mats.at<float>(i, j) * 255);
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
		<< "/" << Epochs << "][" << batch_index << "/" << batches_per_epoch << "] Dis_loss: "<< std::setprecision(4) << d_loss << " | Gen_loss: " << std::setprecision(4) << g_loss << "\r";

}





// Read in the csv file and return file locations and labels as vector of tuples.
auto ReadCsv(std::string& location) -> std::vector<std::tuple<std::string /*file location*/, int64_t /*label*/>> {

    std::fstream in(location, std::ios::in);
    std::string line;
    std::string name;
    std::string label;
    std::vector<std::tuple<std::string, int64_t>> csv;
    int i = 0;
    while (getline(in, line))
    {
        std::stringstream s(line);
        getline(s, name, ',');
        //getline(s, label, ',');
        //std::cout << label;
        csv.push_back(std::make_tuple(name, 0));

        i++;
    }
    //std::cout << csv;
    return csv;
}

