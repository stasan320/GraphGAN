#define _CRT_SECURE_NO_WARNINGS

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

using namespace torch;




struct DCGANGeneratorImpl : nn::Module {
    DCGANGeneratorImpl(int kLatentDim)
        : conv1(nn::ConvTranspose2dOptions(kLatentDim, ImageSize*8, 4).bias(false)),
        batch_norm1(ImageSize*8),

        conv2(nn::ConvTranspose2dOptions(ImageSize*8, ImageSize*4, 4).stride(2).padding(1).bias(false)),
        batch_norm2(ImageSize*4),

        conv3(nn::ConvTranspose2dOptions(ImageSize*4, ImageSize*2, 4).stride(2).padding(1).bias(false)),
        batch_norm3(ImageSize*2),

        conv4(nn::ConvTranspose2dOptions(ImageSize*2, ImageSize, 4).stride(2).padding(1).bias(false)),
        batch_norm4(ImageSize),

        conv5(nn::ConvTranspose2dOptions(ImageSize, ch, 4).stride(2).padding(1).bias(false))
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

    Tensor forwards(Tensor x) {
        x = relu(batch_norm1(conv1(x)));
        x = relu(batch_norm2(conv2(x)));
        x = relu(batch_norm3(conv3(x)));
        x = relu(batch_norm4(conv4(x)));
        x = tanh(conv5(x));
        return x;
    }

    nn::ConvTranspose2d conv1, conv2, conv3, conv4, conv5;
    nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3, batch_norm4;
};

TORCH_MODULE(DCGANGenerator);


torch::Tensor Train(int batch_index, torch::Device device, torch::optim::Adam& discriminator_optimizer, torch::optim::Adam& generator_optimizer, torch::data::datasets::detail::optional_if_t<false, torch::data::Example<>> batch, torch::nn::Sequential discriminator, struct DCGANGenerator generator, torch::Tensor noise) {
    discriminator->zero_grad();
    torch::Tensor real_images = batch.data.to(device);
    torch::Tensor real_labels = torch::empty(batch.data.size(0), device).uniform_(0.8, 1.0);
    torch::Tensor real_output = discriminator->forward(real_images);
    torch::Tensor d_loss_real = binary_cross_entropy(real_output, real_labels);
    d_loss_real.backward();

    // Train discriminator with fake images.

    torch::Tensor fake_images = generator->forwards(noise);
    //Out(fake_images, 1, ImageSize);
    torch::Tensor fake_labels = torch::zeros(batch.data.size(0), device);
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
    manual_seed(time(NULL));

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

    nn::Sequential discriminatorR(
        nn::Conv2d(nn::Conv2dOptions(ch, ImageSize, 4).stride(2).padding(1).bias(false)),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
        nn::Conv2d(nn::Conv2dOptions(ImageSize, ImageSize * 2, 4).stride(2).padding(1).bias(false)),
        nn::BatchNorm2d(ImageSize * 2),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
        nn::Conv2d(nn::Conv2dOptions(ImageSize * 2, ImageSize * 4, 4).stride(2).padding(1).bias(false)),
        nn::BatchNorm2d(ImageSize * 4),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
        nn::Conv2d(nn::Conv2dOptions(ImageSize * 4, ImageSize * 8, 4).stride(2).padding(1).bias(false)),
        nn::BatchNorm2d(ImageSize * 8),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
        nn::Conv2d(nn::Conv2dOptions(ImageSize * 8, 1, 4).stride(1).padding(0).bias(false)),
        nn::Sigmoid());
    nn::Sequential discriminatorG(
        nn::Conv2d(nn::Conv2dOptions(ch, ImageSize, 4).stride(2).padding(1).bias(false)),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
        nn::Conv2d(nn::Conv2dOptions(ImageSize, ImageSize * 2, 4).stride(2).padding(1).bias(false)),
        nn::BatchNorm2d(ImageSize * 2),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
        nn::Conv2d(nn::Conv2dOptions(ImageSize * 2, ImageSize * 4, 4).stride(2).padding(1).bias(false)),
        nn::BatchNorm2d(ImageSize * 4),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
        nn::Conv2d(nn::Conv2dOptions(ImageSize * 4, ImageSize * 8, 4).stride(2).padding(1).bias(false)),
        nn::BatchNorm2d(ImageSize * 8),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
        nn::Conv2d(nn::Conv2dOptions(ImageSize * 8, 1, 4).stride(1).padding(0).bias(false)),
        nn::Sigmoid());
    nn::Sequential discriminatorB(
        nn::Conv2d(nn::Conv2dOptions(ch, ImageSize, 4).stride(2).padding(1).bias(false)),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
        nn::Conv2d(nn::Conv2dOptions(ImageSize, ImageSize * 2, 4).stride(2).padding(1).bias(false)),
        nn::BatchNorm2d(ImageSize * 2),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
        nn::Conv2d(nn::Conv2dOptions(ImageSize * 2, ImageSize * 4, 4).stride(2).padding(1).bias(false)),
        nn::BatchNorm2d(ImageSize * 4),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
        nn::Conv2d(nn::Conv2dOptions(ImageSize * 4, ImageSize * 8, 4).stride(2).padding(1).bias(false)),
        nn::BatchNorm2d(ImageSize * 8),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
        nn::Conv2d(nn::Conv2dOptions(ImageSize * 8, 1, 4).stride(1).padding(0).bias(false)),
        nn::Sigmoid());
    discriminatorR->to(device);
    discriminatorG->to(device);
    discriminatorB->to(device);



    std::string file_names_csv = kCsvFile;
    auto datasetR = FaceDatasetR(file_names_csv).map(data::transforms::Normalize<>(0.5, 0.5)).map(data::transforms::Stack<>());
    auto data_loaderR = data::make_data_loader<data::samplers::RandomSampler>(datasetR, data::DataLoaderOptions().workers(kNumOfWorkers).batch_size(Batch).enforce_ordering(kEnforceOrder));
    auto datasetG = FaceDatasetG(file_names_csv).map(data::transforms::Normalize<>(0.5, 0.5)).map(data::transforms::Stack<>());
    auto data_loaderG = data::make_data_loader<data::samplers::RandomSampler>(datasetG, data::DataLoaderOptions().workers(kNumOfWorkers).batch_size(Batch).enforce_ordering(kEnforceOrder));
    auto datasetB = FaceDatasetB(file_names_csv).map(data::transforms::Normalize<>(0.5, 0.5)).map(data::transforms::Stack<>());                     
    auto data_loaderB = data::make_data_loader<data::samplers::RandomSampler>(datasetB, data::DataLoaderOptions().workers(kNumOfWorkers).batch_size(Batch).enforce_ordering(kEnforceOrder));

    optim::Adam generator_optimizerR(generatorR->parameters(), optim::AdamOptions(kLr).betas(std::make_tuple(kBeta1, kBeta2)));
    optim::Adam discriminator_optimizerR(discriminatorR->parameters(), optim::AdamOptions(kLr).betas(std::make_tuple(kBeta1, kBeta2)));
    optim::Adam generator_optimizerG(generatorG->parameters(), optim::AdamOptions(kLr).betas(std::make_tuple(kBeta1, kBeta2)));
    optim::Adam discriminator_optimizerG(discriminatorG->parameters(), optim::AdamOptions(kLr).betas(std::make_tuple(kBeta1, kBeta2)));
    optim::Adam generator_optimizerB(generatorB->parameters(), optim::AdamOptions(kLr).betas(std::make_tuple(kBeta1, kBeta2)));
    optim::Adam discriminator_optimizerB(discriminatorB->parameters(), optim::AdamOptions(kLr).betas(std::make_tuple(kBeta1, kBeta2)));

    if (1) {
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

        Out(fake_imagesR, fake_imagesG, fake_imagesB, ImageSize);

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
