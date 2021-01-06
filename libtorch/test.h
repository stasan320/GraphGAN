#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <sstream>
#include <fstream>
#include <vector>
#include <tuple>
#include <iostream>

#include <torch/torch.h>
#include <ATen/ATen.h>
#include <torch/script.h>



const int ch = 1;
const int image_size = 64;
int64_t kLatentDim = 100;
const int Batch = 64;
int64_t kNumOfWorkers = 16;
int64_t kNumberOfEpochs = 30000;
int64_t kCheckpointEvery = 20;
int64_t kNumberOfSamplesPerCheckpoint = 64;
int64_t kLogInterval = 10;
double kLr = 2e-4;
double kBeta1 = 0.5;
double kBeta2 = 0.999;



void Out(torch::Tensor R, torch::Tensor G, torch::Tensor B, int image_size) {
    int t = time(NULL);
	//image_size = 64;
    for (int i = 0; i < Batch; i++) {
        cv::Mat matR = cv::Mat(image_size, image_size, CV_32FC1, R[i].data_ptr());
        cv::Mat matG = cv::Mat(image_size, image_size, CV_32FC1, G[i].data_ptr());
        cv::Mat matB = cv::Mat(image_size, image_size, CV_32FC1, B[i].data_ptr());

        cv::Mat out;
        std::vector<cv::Mat> channels = { matR, matG, matB };
        cv::merge(channels, out);

        out.convertTo(out, CV_8UC3, 255);
        cv::imwrite("D:\\Foton\\ngnl_data\\gen_image\\anime\\" + std::to_string(t) + "_" + std::to_string(i) + ".png", out);
        cv::imshow("Out", out);
        cv::waitKey(1);
    }
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


auto ReadCsv() {
    std::string name;
    std::string label;
    std::vector<std::tuple<std::string, int64_t>> csv;
    int l = 64;
    for (int i = 0; i < l; i++) {
        name = "D:\\Foton\\ngnl_data\\training\\help\\anime\\" + std::to_string(i % 60000) + ".png";
        //name = "D:\\DeepFaceLab\\DeepFaceLab_NVIDIA\\workspace\\data_src\\aligned\\0" + std::to_string(i) + ".jpg";
        cv::Mat mat = cv::imread(name);
        if (!mat.data) {
            l++;
            continue;
        }
        csv.push_back(std::make_tuple(name, 1));
    }

    return csv;
};

struct FaceDatasetR : torch::data::Dataset<FaceDatasetR> {
    std::vector<std::tuple<std::string, int64_t>> csv_;
    int k = 0;

    FaceDatasetR() : csv_(ReadCsv()) {

    };

    torch::data::Example<> get(size_t index) override {

        //std::string file_location = "D:\\Foton\\ngnl_data\\training\\help\\anime\\" + std::to_string(k) + ".png";
        std::string file_location = std::get<0>(csv_[k]);
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

        auto tdata = torch::cat({ R }).view({ ch, image_size, image_size }).to(torch::kFloat);
        tdata.permute({ 2, 0, 1 });
        torch::Tensor label_tensor = torch::full({ 1 }, label);

        return { tdata, label_tensor };
    };

    torch::optional<size_t> size() const override {

        return csv_.size();
    };
};



struct FaceDatasetG : torch::data::Dataset<FaceDatasetG> {
    int k = 0;
    std::vector<std::tuple<std::string, int64_t>> csv_;

    FaceDatasetG() : csv_(ReadCsv()) {

    };

    torch::data::Example<> get(size_t index) override {
        //std::string file_location = "D:\\Foton\\ngnl_data\\training\\help\\anime\\" + std::to_string(index % 60000) + ".png";
    //std::string file_location = "D:\\Foton\\ngnl_data\\training\\help\\anime\\" + std::to_string(k) + ".png";
        std::string file_location = std::get<0>(csv_[k]);
        k++;
        if (k > 500) {
            k = 0;
        }
        int64_t label = 1;
        cv::Mat img = cv::imread(file_location);

        cv::resize(img, img, cv::Size(image_size, image_size));


        std::vector<cv::Mat> channels(3);
        cv::split(img, channels);

        //auto R = torch::from_blob(channels[0].ptr(), { ImageSize, ImageSize }, torch::kUInt8);
        auto G = torch::from_blob(channels[1].ptr(), { image_size, image_size }, torch::kUInt8);
        //auto B = torch::from_blob(channels[2].ptr(), { ImageSize, ImageSize }, torch::kUInt8);


        /*cv::imshow("h", img);
        cv::waitKey(1);*/

        auto tdata = torch::cat({ G }).view({ ch, image_size, image_size }).to(torch::kFloat);
        tdata.permute({ 2, 0, 1 });
        torch::Tensor label_tensor = torch::full({ 1 }, label);

        return { tdata, label_tensor };
    };

    torch::optional<size_t> size() const override {

        return csv_.size();
    };
};


struct FaceDatasetB : torch::data::Dataset<FaceDatasetB> {
    int k = 0;
    std::vector<std::tuple<std::string, int64_t>> csv_;

    FaceDatasetB() : csv_(ReadCsv()) {

    };

    torch::data::Example<> get(size_t index) override {

        //std::string file_location = "D:\\Foton\\ngnl_data\\training\\help\\anime\\" + std::to_string(index % 60000) + ".png";
    //std::string file_location = "D:\\Foton\\ngnl_data\\training\\help\\anime\\" + std::to_string(k) + ".png";
        std::string file_location = std::get<0>(csv_[k]);
        k++;
        if (k > 500) {
            k = 0;
        }
        int64_t label = 1;
        cv::Mat img = cv::imread(file_location);

        cv::resize(img, img, cv::Size(image_size, image_size));


        std::vector<cv::Mat> channels(3);
        cv::split(img, channels);

        //auto R = torch::from_blob(channels[0].ptr(), { ImageSize, ImageSize }, torch::kUInt8);
        //auto G = torch::from_blob(channels[1].ptr(), { ImageSize, ImageSize }, torch::kUInt8);
        auto B = torch::from_blob(channels[2].ptr(), { image_size, image_size }, torch::kUInt8);


        /*cv::imshow("h", img);
        cv::waitKey(1);*/

        auto tdata = torch::cat({ B }).view({ ch, image_size, image_size }).to(torch::kFloat);
        tdata.permute({ 2, 0, 1 });
        torch::Tensor label_tensor = torch::full({ 1 }, label);

        return { tdata, label_tensor };
    };

    torch::optional<size_t> size() const override {

        return csv_.size();
    };
};
