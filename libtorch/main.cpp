#define _CRT_SECURE_NO_WARNINGS

#include <torch/torch.h>

#include <cmath>
#include <cstdio>
#include <iostream>
#include "test.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>


const int SizeInputG = 100;
const int Batch = 64;
const int Epochs = 30;
const char* path = "D:\\Foton\\xz";
const int Saving = 200;

//const bool Loading = true;
const bool Loading = false;

using namespace torch;

struct DCGANGeneratorImpl:
    nn::Module {
    DCGANGeneratorImpl(int SizeInputG):
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

TORCH_MODULE(DCGANGenerator);

int main() {
    /*cv::Mat mat = cv::Mat(28, 28, CV_32FC1);
    cv::Mat image(28, 28, CV_8UC1);
       
    image = cv::imread("D:\\Foton\\ngnl_data\\1.png");
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            mat.at<float>(i, j) = -1 + 2*(float)image.at<uchar>(i, j) / 255;
        }
    }

    //cv::imshow("Out", image);
    cv::waitKey(1);

    torch::Tensor real_images = torch::from_blob(mat.data, { mat.rows, mat.cols});
    //std::cout << real_images;

    Out(real_images, 0);
    cv::waitKey(1000);*/

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

    auto dataset = torch::data::datasets::MNIST(path).map(torch::data::transforms::Normalize<>(0.5, 0.5)).map(torch::data::transforms::Stack<>());
    const int batches_per_epoch = std::ceil(dataset.size().value() / static_cast<double>(Batch));

    auto data_loader = torch::data::make_data_loader(std::move(dataset), torch::data::DataLoaderOptions().batch_size(Batch).workers(2));

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
        for (torch::data::Example<>& batch : *data_loader) {
            discriminator->zero_grad();
            //torch::Tensor real_images = batch.data.to(device);

            torch::Tensor tensor_image[Batch];


            for (int k = 0; k < batch.data.size(0); k++) {
                cv::Mat image = cv::imread("D:\\Foton\\ngnl_data\\training\\" + std::to_string(rand() % 10) + "\\1 (" + std::to_string(rand() % 5000) + ").png", CV_8UC1);
                cv::Mat im(28, 28, CV_32FC1);
                //cv::threshold(image, image, -1, 1, CV_32FC1);
                for (int i = 0; i < 28; i++) {
                    for (int j = 0; j < 28; j++) {
                        //std::cout << (int)image.at<uchar>(i, j) << std::endl;
                        im.at<float>(i, j) = -1 + (float)2 * image.at<uchar>(i, j) / 255;

                        //std::cout << (float)im.at<float>(i, j) << std::endl;
                    }
                }

                //cv::imshow("Ou5t", im);


                torch::Tensor tensor = torch::from_blob(im.data, { 28, 28, 1 });
                tensor_image[k] = tensor;
               /* cv::Mat mat = cv::Mat(28, 28, CV_32FC1, tensor.data_ptr());
                cv::imshow("Out", mat);
                cv::waitKey(1);*/

                //cv::Mat mat = cv::Mat(28, 28, CV_32FC1, input_tensor.data_ptr());

                //cv::imshow("Out", img);
                //cv::waitKey(100);
                //tensor_image[i] = torch::from_blob(img.data, { img.rows, img.cols, 1}, at::kFloat);
                //tensor_image = tensor_image.permute({ 2,0,1 });
            }

            torch::Tensor real_images = torch::from_blob(tensor_image, { 28, 28, 1 , Batch});


            //tensor_image = torch::from_blob(inputs.data(), {28, 28, 1, Batch});
            /*for (int i = 0; i < batch.data.size(0); i++) {
                
                cv::Mat mat = cv::Mat(28, 28, CV_32FC1, tensor_image[i].data_ptr());
                std::cout << i;
                cv::imshow("Out", mat);
                cv::waitKey(100);
            }*/

            //cv::Mat mat = cv::Mat(28, 28, CV_32FC1, real_images.data_ptr());
            //cv::imshow("Out", mat);
            //cv::waitKey(1);

            torch::Tensor real_labels = torch::empty(batch.data.size(0), device).uniform_(0.8, 1.0);            //uniform нормализует дикий рандом, судя по всему это тензор прав результатов для настоящих изобр
            //std::cout << real_labels << std::endl;
            torch::Tensor real_output = discriminator->forward(real_images);
            torch::Tensor d_loss_real = torch::binary_cross_entropy(real_output, real_labels);
            d_loss_real.backward();

            torch::Tensor noise = torch::randn({ batch.data.size(0), SizeInputG, 1, 1 }, device);
            torch::Tensor fake_images = generator->SumReTa(noise);
            torch::Tensor fake_labels = torch::zeros(batch.data.size(0), device);             //нулевые значения для генератора
            torch::Tensor fake_output = discriminator->forward(fake_images.detach());

            Out(fake_images, batch_index);
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
