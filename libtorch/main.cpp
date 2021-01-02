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

using namespace torch;


int ch = 33;
// Reshaped Image Size
const int64_t ImageSize = 64;

// The size of the noise vector fed to the generator.
const int64_t kLatentDim = 100;

// The batch size for training.
const int64_t kBatchSize = 64;

// Number of workers
const int64_t kNumOfWorkers = 16;

// Enforce ordering
const bool kEnforceOrder = false;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 30000;

// Where to find the CSV with file locations.
const string kCsvFile = "../file_names.csv";

// After how many batches to create a new checkpoint periodically.
const int64_t kCheckpointEvery = 50;

// How many images to sample at every checkpoint.
const int64_t kNumberOfSamplesPerCheckpoint = 64;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

// Learning Rate
const double kLr = 2e-4;

// Beta1
const double kBeta1 = 0.5;

// Beta2
const double kBeta2 = 0.999;

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

auto ReadCsv(std::string& location) {
    std::fstream in(location, std::ios::in);
    std::string line;
    std::string name;
    std::string label;
    std::vector<std::tuple<std::string, int64_t>> csv;

    for (int i = 0; i < 5000; i++) {
        name = "D:\\Foton\\ngnl_data\\data\\data\\cropped\\1 (" + std::to_string(i % 5000 + 1) + ").jpg";
        csv.push_back(std::make_tuple(name, 1));
    }

    return csv;
};

struct FaceDataset : torch::data::Dataset<FaceDataset>
{

    std::vector<std::tuple<std::string /*file location*/, int64_t /*label*/>> csv_;

    FaceDataset(std::string& file_names_csv)
        // Load csv file with file locations and labels.
        : csv_(ReadCsv(file_names_csv)) {

    };

    // Override the get method to load custom data.
    torch::data::Example<> get(size_t index) override {

        std::string file_location = "D:\\Foton\\ngnl_data\\training\\help\\anime\\" + std::to_string(0) + ".png";
        //std::string file_location = "D:\\Foton\\ngnl_data\\training\\help\\0\\" + std::to_string(index % 5000) + ".png";
        int64_t label = 1;

        // Load image with OpenCV.
        cv::Mat img = cv::imread(file_location);

        cv::resize(img, img, cv::Size(ImageSize, ImageSize));


        std::vector<cv::Mat> channels(3);
        cv::split(img, channels);

        auto R = torch::from_blob(
            channels[2].ptr(),
            { ImageSize, ImageSize },
            torch::kUInt8);
        auto G = torch::from_blob(
            channels[1].ptr(),
            { ImageSize, ImageSize },
            torch::kUInt8);
        auto B = torch::from_blob(
            channels[0].ptr(),
            { ImageSize, ImageSize },
            torch::kUInt8);

        /*cv::imshow("h", channels[0]);
        cv::waitKey(100);*/

        auto tdata = torch::cat({ R, G, B }).view({ ch, ImageSize, ImageSize }).to(torch::kFloat);
        tdata.permute({ 2, 0, 1 });

        // Convert the image and label to a tensor.
        // Here we need to clone the data, as from_blob does not change the ownership of the underlying memory,
        // which, therefore, still belongs to OpenCV. If we did not clone the data at this point, the memory
        // would be deallocated after leaving the scope of this get method, which results in undefined behavior.
        Tensor img_tensor = torch::from_blob(img.data, { img.rows, img.cols, ch }, torch::kByte).clone();
        img_tensor = img_tensor.permute({ 2, 0, 1 }); // convert to CxHxW
        Tensor label_tensor = torch::full({ 1 }, label);

        return { tdata, label_tensor };
    };

    // Override the size method to infer the size of the data set.
    torch::optional<size_t> size() const override {

        return csv_.size();
    };
};

int main() {
    manual_seed(1);

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device = torch::Device(torch::kCUDA);
    }

    DCGANGenerator generator(kLatentDim);
    generator->to(device);

    nn::Sequential discriminator(
        // Layer 1
        nn::Conv2d(nn::Conv2dOptions(ch, ImageSize, 4).stride(2).padding(1).bias(false)),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
        // Layer 2
        nn::Conv2d(nn::Conv2dOptions(ImageSize, ImageSize*2, 4).stride(2).padding(1).bias(false)),
        nn::BatchNorm2d(ImageSize*2),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
        // Layer 3
        nn::Conv2d(nn::Conv2dOptions(ImageSize*2, ImageSize*4, 4).stride(2).padding(1).bias(false)),
        nn::BatchNorm2d(ImageSize*4),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
        // Layer 4
        nn::Conv2d(nn::Conv2dOptions(ImageSize*4, ImageSize*8, 4).stride(2).padding(1).bias(false)),
        nn::BatchNorm2d(ImageSize*8),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
        // Layer 5
        nn::Conv2d(nn::Conv2dOptions(ImageSize*8, 1, 4).stride(1).padding(0).bias(false)),
        nn::Sigmoid());
    discriminator->to(device);



    std::string file_names_csv = kCsvFile;
    auto dataset = FaceDataset(file_names_csv).map(data::transforms::Normalize<>(0.5, 0.5)).map(data::transforms::Stack<>());

    const int64_t batches_per_epoch = std::ceil(dataset.size().value() / static_cast<double>(kBatchSize));

    auto data_loader = data::make_data_loader<data::samplers::RandomSampler>(dataset, data::DataLoaderOptions().workers(kNumOfWorkers).batch_size(kBatchSize).enforce_ordering(kEnforceOrder));

    optim::Adam generator_optimizer(generator->parameters(), optim::AdamOptions(kLr).betas(std::make_tuple(kBeta1, kBeta2)));
    optim::Adam discriminator_optimizer(discriminator->parameters(), optim::AdamOptions(kLr).betas(std::make_tuple(kBeta1, kBeta2)));


    if (false) {
        torch::load(generator, "D:\\Foton\\xz\\dat\\generator-checkpoint.pt");
        torch::load(generator_optimizer, "D:\\Foton\\xz\\dat\\generator-optimizer-checkpoint.pt");
        torch::load(discriminator, "D:\\Foton\\xz\\dat\\discriminator-checkpoint.pt");
        torch::load(discriminator_optimizer, "D:\\Foton\\xz\\dat\\discriminator-optimizer-checkpoint.pt");
    }



    int64_t checkpoint_counter = 1;
    for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
        int64_t batch_index = 0;
        for (auto& batch : *data_loader) {
            // Train discriminator with real images.
            discriminator->zero_grad();
            Tensor real_images = batch.data.to(device);
            Tensor real_labels = torch::empty(batch.data.size(0), device).uniform_(0.8, 1.0);
            Tensor real_output = discriminator->forward(real_images);
            Tensor d_loss_real = binary_cross_entropy(real_output, real_labels);
            d_loss_real.backward();

            // Train discriminator with fake images.
            Tensor noise = torch::randn({ batch.data.size(0), kLatentDim, 1, 1 }, device);
            Tensor fake_images = generator->forwards(noise);
            Out(fake_images, 1, ImageSize);
            Tensor fake_labels = torch::zeros(batch.data.size(0), device);
            Tensor fake_output = discriminator->forward(fake_images.detach());
            Tensor d_loss_fake = torch::binary_cross_entropy(fake_output, fake_labels);
            d_loss_fake.backward();

            Tensor d_loss = d_loss_real + d_loss_fake;
            discriminator_optimizer.step();

            // Train generator.
            generator->zero_grad();
            fake_labels.fill_(1);
            fake_output = discriminator->forward(fake_images);
            Tensor g_loss = torch::binary_cross_entropy(fake_output, fake_labels);
            g_loss.backward();
            generator_optimizer.step();
            batch_index++;
            if (batch_index % kLogInterval == 0) {
                std::printf(
                    "\r[%2ld/%2ld][%3ld/%3ld] D_loss: %.4f | G_loss: %.4f\n",
                    epoch,
                    kNumberOfEpochs,
                    batch_index,
                    batches_per_epoch,
                    d_loss.item<float>(),
                    g_loss.item<float>());
            }
            if (batch_index % kCheckpointEvery == 0) {
                // Checkpoint the model and optimizer state.
                torch::save(generator, "D:\\Foton\\xz\\dat\\generator-checkpoint.pt");
                torch::save(generator_optimizer, "D:\\Foton\\xz\\dat\\generator-optimizer-checkpoint.pt");
                torch::save(discriminator, "D:\\Foton\\xz\\dat\\discriminator-checkpoint.pt");
                torch::save(discriminator_optimizer, "D:\\Foton\\xz\\dat\\discriminator-optimizer-checkpoint.pt");
                // Sample the generator and save the images.
                Tensor samples = generator->forwards(torch::randn({ kNumberOfSamplesPerCheckpoint, kLatentDim, 1, 1 }, device));
                torch::save((samples + 1.0) / 2.0, torch::str("dcgan-sample-", checkpoint_counter, ".pt"));
                std::cout << "\n-> checkpoint " << ++checkpoint_counter << '\n';
            }
        }
    }

    std::cout << "Training complete!" << std::endl;
}
