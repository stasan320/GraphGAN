#define _CRT_SECURE_NO_WARNINGS
#pragma comment(lib, "torch_cpu.lib")
#pragma comment(lib, "c10.lib")
#pragma comment(lib, "c10d.lib")
#pragma comment(lib, "cpuinfo.lib")
#pragma comment(lib, "caffe2_nvrtc.lib")
#pragma comment(lib, "torch_cuda.lib")
#pragma comment(lib, "torch_cpu.lib")
#pragma comment(lib, "torch.lib")
#pragma comment(lib, "c10_cuda.lib")
#pragma comment(lib, "asmjit.lib")
#pragma comment(lib, "opencv_world3412.lib")
#pragma comment(lib, "opencv_world3412d.lib")


#include <torch/torch.h>
#include <ATen/ATen.h>

#include <cmath>
#include <cstdio>
#include <iostream>
#include "test.h"
#include "constant.h"
#include <iostream>
#include <vector>
#include <tuple>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>




struct DCGANGeneratorImpl : torch::nn::Module {
    DCGANGeneratorImpl(int kLatentDim)
        : conv1(torch::nn::ConvTranspose2dOptions(kLatentDim, image_size*8, 4).bias(false)),
        batch_norm1(image_size*8),

        conv2(torch::nn::ConvTranspose2dOptions(image_size*8, image_size*4, 4).stride(2).padding(1).bias(false)),
        batch_norm2(image_size*4),

        conv3(torch::nn::ConvTranspose2dOptions(image_size*4, image_size*2, 4).stride(2).padding(1).bias(false)),
        batch_norm3(image_size*2),

        conv4(torch::nn::ConvTranspose2dOptions(image_size*2, image_size, 4).stride(2).padding(1).bias(false)),
        batch_norm4(image_size),

        conv5(torch::nn::ConvTranspose2dOptions(image_size, ch, 4).stride(2).padding(1).bias(false))
    {
        // register_module() is needed if we want to use the parameters() method later on
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        register_module("conv4", conv4);
        register_module("conv5", conv5);
        register_module("batch_norm1", batch_norm1);
        register_module("batch_norm2", batch_norm2);
        register_module("batch_norm3", batch_norm3);
        register_module("batch_norm4", batch_norm4);
    }

    torch::Tensor forwards(torch::Tensor x) {
        x = torch::relu(batch_norm1(conv1(x)));
        x = torch::relu(batch_norm2(conv2(x)));
        x = torch::relu(batch_norm3(conv3(x)));
        x = torch::relu(batch_norm4(conv4(x)));
        x = torch::tanh(conv5(x));
        return x;
    }

    torch::nn::ConvTranspose2d conv1, conv2, conv3, conv4, conv5;
    torch::nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3, batch_norm4;
};

TORCH_MODULE(DCGANGenerator);


torch::Tensor Train(int batch_index, 
                    torch::Device device, 
                    torch::optim::Adam& discriminator_optimizer, 
                    torch::optim::Adam& generator_optimizer, 
                    torch::data::datasets::detail::optional_if_t<false, 
                    torch::data::Example<>> batch, 
                    torch::nn::Sequential discriminator, 
                    struct DCGANGenerator generator, 
                    torch::Tensor noise) {
    discriminator->zero_grad();
    torch::Tensor real_images = batch.data.to(device);
    torch::Tensor real_labels = torch::empty(Batch, device).uniform_(0.8, 1.0);
    torch::Tensor real_output = discriminator->forward(real_images);
    torch::Tensor d_loss_real = binary_cross_entropy(real_output, real_labels);
    d_loss_real.backward();

    // Train discriminator with fake images.

    torch::Tensor fake_images = generator->forwards(noise);
    //Out(fake_images, 1, ImageSize);
    torch::Tensor fake_labels = torch::zeros(Batch, device);
    torch::Tensor fake_output = discriminator->forward(fake_images.detach());
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

    return fake_images;
}



int main() {
    torch::manual_seed(time(NULL));

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device = torch::Device(torch::kCUDA);
    }

    DCGANGenerator generatorR(kLatentDim);
    DCGANGenerator generatorG(kLatentDim);
    DCGANGenerator generatorB(kLatentDim);
    generatorR->to(device);
    generatorG->to(device);
    generatorB->to(device);

    torch::nn::Sequential discriminatorR(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(ch, image_size, 4).stride(2).padding(1).bias(false)),
        torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(image_size, image_size * 2, 4).stride(2).padding(1).bias(false)),
        torch::nn::BatchNorm2d(image_size * 2),
        torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(image_size * 2, image_size * 4, 4).stride(2).padding(1).bias(false)),
        torch::nn::BatchNorm2d(image_size * 4),
        torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(image_size * 4, image_size * 8, 4).stride(2).padding(1).bias(false)),
        torch::nn::BatchNorm2d(image_size * 8),
        torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(image_size * 8, 1, 4).stride(1).padding(0).bias(false)),
        torch::nn::Sigmoid());
    torch::nn::Sequential discriminatorG(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(ch, image_size, 4).stride(2).padding(1).bias(false)),
        torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(image_size, image_size * 2, 4).stride(2).padding(1).bias(false)),
        torch::nn::BatchNorm2d(image_size * 2),
        torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(image_size * 2, image_size * 4, 4).stride(2).padding(1).bias(false)),
        torch::nn::BatchNorm2d(image_size * 4),
        torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(image_size * 4, image_size * 8, 4).stride(2).padding(1).bias(false)),
        torch::nn::BatchNorm2d(image_size * 8),
        torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(image_size * 8, 1, 4).stride(1).padding(0).bias(false)),
        torch::nn::Sigmoid());
    torch::nn::Sequential discriminatorB(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(ch, image_size, 4).stride(2).padding(1).bias(false)),
        torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(image_size, image_size * 2, 4).stride(2).padding(1).bias(false)),
        torch::nn::BatchNorm2d(image_size * 2),
        torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(image_size * 2, image_size * 4, 4).stride(2).padding(1).bias(false)),
        torch::nn::BatchNorm2d(image_size * 4),
        torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(image_size * 4, image_size * 8, 4).stride(2).padding(1).bias(false)),
        torch::nn::BatchNorm2d(image_size * 8),
        torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(image_size * 8, 1, 4).stride(1).padding(0).bias(false)),
        torch::nn::Sigmoid());
    discriminatorR->to(device);
    discriminatorG->to(device);
    discriminatorB->to(device);



    auto datasetR = FaceDatasetR().map(torch::data::transforms::Normalize<>(0.5, 0.5)).map(torch::data::transforms::Stack<>());
    auto data_loaderR = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(datasetR, torch::data::DataLoaderOptions().workers(kNumOfWorkers).batch_size(Batch).enforce_ordering(false));
    auto datasetG = FaceDatasetG().map(torch::data::transforms::Normalize<>(0.5, 0.5)).map(torch::data::transforms::Stack<>());
    auto data_loaderG = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(datasetG, torch::data::DataLoaderOptions().workers(kNumOfWorkers).batch_size(Batch).enforce_ordering(false));
    auto datasetB = FaceDatasetB().map(torch::data::transforms::Normalize<>(0.5, 0.5)).map(torch::data::transforms::Stack<>());
    auto data_loaderB = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(datasetB, torch::data::DataLoaderOptions().workers(kNumOfWorkers).batch_size(Batch).enforce_ordering(false));

    torch::optim::Adam generator_optimizerR(generatorR->parameters(), torch::optim::AdamOptions(kLr).betas(std::make_tuple(kBeta1, kBeta2)));
    torch::optim::Adam discriminator_optimizerR(discriminatorR->parameters(), torch::optim::AdamOptions(kLr).betas(std::make_tuple(kBeta1, kBeta2)));
    torch::optim::Adam generator_optimizerG(generatorG->parameters(), torch::optim::AdamOptions(kLr).betas(std::make_tuple(kBeta1, kBeta2)));
    torch::optim::Adam discriminator_optimizerG(discriminatorG->parameters(), torch::optim::AdamOptions(kLr).betas(std::make_tuple(kBeta1, kBeta2)));
    torch::optim::Adam generator_optimizerB(generatorB->parameters(), torch::optim::AdamOptions(kLr).betas(std::make_tuple(kBeta1, kBeta2)));
    torch::optim::Adam discriminator_optimizerB(discriminatorB->parameters(), torch::optim::AdamOptions(kLr).betas(std::make_tuple(kBeta1, kBeta2)));

    if (false) {
        torch::load(generatorR, "D:\\Foton\\xz\\dat\\generatorR-checkpoint.pt");
        torch::load(generator_optimizerR, "D:\\Foton\\xz\\dat\\generator-optimizerR-checkpoint.pt");
        torch::load(discriminatorR, "D:\\Foton\\xz\\dat\\discriminatorR-checkpoint.pt");
        torch::load(discriminator_optimizerR, "D:\\Foton\\xz\\dat\\discriminator-optimizerR-checkpoint.pt");

        torch::load(generatorG, "D:\\Foton\\xz\\dat\\generatorG-checkpoint.pt");
        torch::load(generator_optimizerG, "D:\\Foton\\xz\\dat\\generator-optimizerG-checkpoint.pt");
        torch::load(discriminatorG, "D:\\Foton\\xz\\dat\\discriminatorG-checkpoint.pt");
        torch::load(discriminator_optimizerG, "D:\\Foton\\xz\\dat\\discriminator-optimizerG-checkpoint.pt");

        torch::load(generatorB, "D:\\Foton\\xz\\dat\\generatorB-checkpoint.pt");
        torch::load(generator_optimizerB, "D:\\Foton\\xz\\dat\\generator-optimizerB-checkpoint.pt");
        torch::load(discriminatorB, "D:\\Foton\\xz\\dat\\discriminatorB-checkpoint.pt");
        torch::load(discriminator_optimizerB, "D:\\Foton\\xz\\dat\\discriminator-optimizerB-checkpoint.pt");
    }


    int64_t checkpoint_counter = 1;
    int64_t batch_index = 0;
    for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
        torch::Tensor fake_imagesR, fake_imagesG, fake_imagesB;
        torch::Tensor noise = torch::randn({ Batch, kLatentDim, 1, 1 }, device);

        for (auto& batch : *data_loaderR) {
            fake_imagesR = Train(batch_index, device, discriminator_optimizerR, generator_optimizerR, batch, discriminatorR, generatorR, noise);
        }
        for (auto& batch : *data_loaderG) {
            fake_imagesG = Train(batch_index, device, discriminator_optimizerG, generator_optimizerG, batch, discriminatorG, generatorG, noise);
        }
        for (auto& batch : *data_loaderB) {
            fake_imagesB = Train(batch_index, device, discriminator_optimizerB, generator_optimizerB, batch, discriminatorB, generatorB, noise);
            batch_index++;
        }

        Out(fake_imagesR, fake_imagesG, fake_imagesB, image_size);

        if (batch_index % kCheckpointEvery == 0) {
            torch::save(generatorR, "D:\\Foton\\xz\\dat\\generatorR-checkpoint.pt");
            torch::save(generator_optimizerR, "D:\\Foton\\xz\\dat\\generator-optimizerR-checkpoint.pt");
            torch::save(discriminatorR, "D:\\Foton\\xz\\dat\\discriminatorR-checkpoint.pt");
            torch::save(discriminator_optimizerR, "D:\\Foton\\xz\\dat\\discriminator-optimizerR-checkpoint.pt");

            torch::save(generatorG, "D:\\Foton\\xz\\dat\\generatorG-checkpoint.pt");
            torch::save(generator_optimizerG, "D:\\Foton\\xz\\dat\\generator-optimizerG-checkpoint.pt");
            torch::save(discriminatorG, "D:\\Foton\\xz\\dat\\discriminatorG-checkpoint.pt");
            torch::save(discriminator_optimizerG, "D:\\Foton\\xz\\dat\\discriminator-optimizerG-checkpoint.pt");

            torch::save(generatorB, "D:\\Foton\\xz\\dat\\generatorB-checkpoint.pt");
            torch::save(generator_optimizerB, "D:\\Foton\\xz\\dat\\generator-optimizerB-checkpoint.pt");
            torch::save(discriminatorB, "D:\\Foton\\xz\\dat\\discriminatorB-checkpoint.pt");
            torch::save(discriminator_optimizerB, "D:\\Foton\\xz\\dat\\discriminator-optimizerB-checkpoint.pt");
            std::cout << "Saving RGB model" << std::endl;
        }
    }

    std::cout << "Training complete!" << std::endl;
}
