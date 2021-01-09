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


const int image_size = 64;

int d[6] = { image_size*image_size, image_size * 8, image_size * 4, image_size * 2, image_size, 1 };
int g[6] = { 100, image_size, image_size * 2, image_size * 4, image_size * 8, image_size * image_size };
const int Workers = 16;
const int channel = 1;

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



auto ReadCsv() {
    std::string name;
    std::string label;
    std::vector<std::tuple<std::string, int64_t>> csv;
    for (int i = 0; i < Batch; i++) {
        csv.push_back(std::make_tuple(std::to_string(i), 1));
    }

    return csv;
};

struct FaceDatasetR : torch::data::Dataset<FaceDatasetR> {
    std::vector<std::tuple<std::string, int64_t>> csv_;
    int k = 0;

    FaceDatasetR() : csv_(ReadCsv()) {

    };

    torch::data::Example<> get(size_t index) override {

        std::string file_location = "D:\\Foton\\ngnl_data\\training\\help\\" + std::to_string(index % 10) + "\\" + std::to_string(k % 60000) + ".png";
        //std::string file_location = std::get<0>(csv_[k]);
        k++;
        if (k > 500) {
            k = 0;
        }
        int64_t label = 1;
        cv::Mat img = cv::imread(file_location);

        cv::resize(img, img, cv::Size(image_size, image_size));


        std::vector<cv::Mat> channels(3);
        cv::split(img, channels);

        auto R = torch::from_blob(channels[0].ptr(), { image_size, image_size }, torch::kUInt8);
        //auto G = torch::from_blob(channels[1].ptr(), { ImageSize, ImageSize }, torch::kUInt8);
        //auto B = torch::from_blob(channels[2].ptr(), { ImageSize, ImageSize }, torch::kUInt8);


        /*cv::imshow("h", img);
        cv::waitKey(1);*/

        auto tdata = torch::cat({ R }).view({ channel, image_size, image_size }).to(torch::kFloat);
        tdata.permute({ 2, 0, 1 });
        torch::Tensor label_tensor = torch::full({ 1 }, label);

        return { tdata, label_tensor };
    };

    torch::optional<size_t> size() const override {

        return csv_.size();
    };
};
