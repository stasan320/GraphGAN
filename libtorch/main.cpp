#define _CRT_SECURE_NO_WARNINGS

#include <torch/torch.h>
#include <ATen/ATen.h>

#include <cmath>
#include <cstdio>
#include <iostream>
#include "test.h"
#include <iostream>
#include <vector>
#include <tuple>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>





const int SizeInputG = 100;
const int Batch = 64;
const int Epochs = 30;
const char* path = "D:\\Foton\\data\\d";
const int Saving = 200;
const int LData = 5000;
const int image_size = 28;

//const bool Loading = true;
const bool Loading = false;

using namespace torch;



struct DCGANGeneratorImpl :
    nn::Module {
    DCGANGeneratorImpl(int SizeInputG) :
        conv1(nn::ConvTranspose2dOptions(SizeInputG, 256, 4).bias(false)), batch_norm1(256),
        conv2(nn::ConvTranspose2dOptions(256, 200, 3).stride(2).padding(1).bias(false)), batch_norm2(200),
        conv3(nn::ConvTranspose2dOptions(200, 64, 4).stride(2).padding(1).bias(false)), batch_norm3(64),
        conv4(nn::ConvTranspose2dOptions(64, 1, 4).stride(2).padding(1).bias(false))
    {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        register_module("conv4", conv4);
        register_module("batch_norm1", batch_norm1);
        register_module("batch_norm2", batch_norm2);
        register_module("batch_norm3", batch_norm3);
    }

    torch::Tensor SumReTa(torch::Tensor x) {
        x = torch::relu(batch_norm1(conv1(x)));
        x = torch::relu(batch_norm2(conv2(x)));
        x = torch::relu(batch_norm3(conv3(x)));
        x = torch::tanh(conv4(x));
        return x;
    }

    nn::ConvTranspose2d conv1, conv2, conv3, conv4;
    nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3;
};

struct Options {
    int image_size = 28;
    size_t train_batch_size = 64;
    size_t test_batch_size = 200;
    size_t iterations = 10;
    size_t log_interval = 20;
    // path must end in delimiter
    std::string datasetPath = "D:\\Foton\\ngnl_data\\training\\help\\0\\";
    std::string infoFilePath = "D:\\Foton\\data\\info.txt";
    torch::DeviceType device = torch::kCPU;
};



using Data = std::vector<std::pair<std::string, long>>;
static Options options;


std::pair<Data, Data> readInfo() {
    Data train, test;

    std::ifstream stream(options.infoFilePath);
    assert(stream.is_open());

    long label = 1;
    std::string path, type;
    for (int i = 0; i < 5000; i++) {
        path = std::to_string(i) + ".png";
        train.push_back(std::make_pair(path, label));
    }
    std::random_shuffle(train.begin(), train.end());
    std::random_shuffle(test.begin(), test.end());
    return std::make_pair(train, test);
}


class CustomDataset : public torch::data::datasets::Dataset<CustomDataset> {
    using Example = torch::data::Example<>;

    Data data;

public:
    CustomDataset(const Data& data) : data(data) {}

    Example get(size_t index) {
        std::string path = options.datasetPath + data[index].first;
        auto mat = cv::imread(path);
        //cv::Mat im = cv::imread("D:\\Foton\\ngnl_data\\img_align_celeba\\000001.jpg");
        cv::resize(mat, mat, { 64, 64 });
        //cv::imshow("Out", mat);
        //cv::waitKey(10);
        //std::cout << index;
        /*cv::imshow("Out", mat);
        cv::waitKey(1);*/
        assert(!mat.empty());

        cv::resize(mat, mat, cv::Size(image_size, image_size));
        std::vector<cv::Mat> channels(3);
        cv::split(mat, channels);

        /*auto R = torch::from_blob(
            channels[2].ptr(),
            { options.image_size, options.image_size },
            torch::kUInt8);
        auto G = torch::from_blob(
            channels[1].ptr(),
            { options.image_size, options.image_size },
            torch::kUInt8);*/
        auto B = torch::from_blob(
            channels[0].ptr(),
            { image_size, image_size },
            torch::kUInt8);

        auto tdata = torch::cat({ B }).view({ 1, image_size, image_size }).to(torch::kFloat);
        auto tlabel = torch::from_blob(&data[index].second, { 1 }, torch::kLong);
        return { tdata, tlabel };
    }

    torch::optional<size_t> size() const {
        return data.size();
    }
};

TORCH_MODULE(DCGANGenerator);




int main() {

    time_t t = 0;
    torch::manual_seed(1);
    torch::Device device(torch::kCPU);

    DCGANGenerator generator(SizeInputG);
    generator->to(device);

    nn::Sequential discriminator(
        nn::Conv2d(nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(false)),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),

        nn::Conv2d(nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).bias(false)),
        nn::BatchNorm2d(128), nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),

        nn::Conv2d(nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).bias(false)),
        nn::BatchNorm2d(256),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),

        nn::Conv2d(nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).bias(false)),
        nn::Sigmoid());
    discriminator->to(device);

    auto data = readInfo();
    auto train_set = CustomDataset(data.first).map(torch::data::transforms::Stack<>());
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_set), Batch);

    torch::optim::Adam generator_optimizer(generator->parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple(0.5, 0.5)));
    torch::optim::Adam discriminator_optimizer(discriminator->parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple(0.5, 0.5)));




    if (Loading) {
        torch::load(generator, "D:\\Foton\\xz\\dat\\generator-checkpoint.pt");
        torch::load(generator_optimizer, "D:\\Foton\\xz\\dat\\generator-optimizer-checkpoint.pt");
        torch::load(discriminator, "D:\\Foton\\xz\\dat\\discriminator-checkpoint.pt");
        torch::load(discriminator_optimizer, "D:\\Foton\\xz\\dat\\discriminator-optimizer-checkpoint.pt");
    }

    int checkpoint_counter = 1;
    for (int epoch = 0; epoch < Epochs; epoch++) {
        int batch_index = 0;
        for (auto& batch : *data_loader) {
            discriminator->zero_grad();
            torch::Tensor real_images = batch.data.to(device);
            torch::Tensor real_labels = torch::empty(batch.data.size(0), device).uniform_(0.8, 1.0);            //uniform нормализует дикий рандом, судя по всему это тензор прав результатов для настоящих изобр


            //std::cout << real_labels << std::endl;
            torch::Tensor real_output = discriminator->forward(real_images);
            torch::Tensor d_loss_real = torch::binary_cross_entropy(real_output, real_labels);
            d_loss_real.backward();
            torch::Tensor noise = torch::randn({ batch.data.size(0), SizeInputG, 1, 1 }, device);
            torch::Tensor fake_images = generator->SumReTa(noise);
            torch::Tensor fake_labels = torch::zeros(batch.data.size(0), device);             //нулевые значения для генератора
            torch::Tensor fake_output = discriminator->forward(fake_images.detach());

            Out(fake_images, batch_index, image_size);
            torch::Tensor d_loss_fake = torch::binary_cross_entropy(fake_output, fake_labels);
            d_loss_fake.backward();

            torch::Tensor d_loss = d_loss_real + d_loss_fake;
            discriminator_optimizer.step();

            // Train generator.
            generator->zero_grad();
            fake_labels.fill_(1);
            fake_output = discriminator->forward(fake_images);
            torch::Tensor g_loss = torch::binary_cross_entropy(fake_output, fake_labels);
            g_loss.backward();
            generator_optimizer.step();
            batch_index++;

            if (time(NULL) > t) {
                t = time(NULL);
                //ConsoleData(epoch, Epochs, batch_index, batches_per_epoch, d_loss.item<float>(), g_loss.item<float>());
            }

            if (batch_index % Saving == 0) {
                torch::save(generator, "D:\\Foton\\xz\\dat\\generator-checkpoint.pt");
                torch::save(generator_optimizer, "D:\\Foton\\xz\\dat\\generator-optimizer-checkpoint.pt");
                torch::save(discriminator, "D:\\Foton\\xz\\dat\\discriminator-checkpoint.pt");
                torch::save(discriminator_optimizer, "D:\\Foton\\xz\\dat\\discriminator-optimizer-checkpoint.pt");

                torch::Tensor samples = generator->SumReTa(torch::randn({ 10, SizeInputG, 1, 1 }, device));
                torch::save((samples + 1.0) / 2.0, torch::str("D:\\Foton\\xz\\dat\\dcgan-sample-", checkpoint_counter, ".pt"));
                std::cout << "\n-> checkpoint " << checkpoint_counter++ << '\n';
            }

        }
    }

    std::cout << "Training complete!" << std::endl;
}
